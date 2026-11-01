"""
FraudLens SQLite Database Manager
Provides safe, local persistence for analyzed transactions, audit logs,
statistics, and analytics aggregation without external database dependencies.
"""

import os
import sqlite3
import zoneinfo
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Tuple

IST_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")


@contextmanager
def get_db(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that guarantees the connection is properly closed."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Initialize the transactions table and indexes if they do not exist."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                channel TEXT NOT NULL,
                merchant_category TEXT NOT NULL,
                country TEXT NOT NULL,
                hours_since_prev_txn REAL NOT NULL,
                prediction TEXT NOT NULL,
                label TEXT NOT NULL,
                is_fraud INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                currency TEXT DEFAULT NULL
            );
            """
        )
        # Idempotent migration for existing database schemas:
        cursor.execute("PRAGMA table_info(transactions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        if "currency" not in existing_cols:
            cursor.execute("ALTER TABLE transactions ADD COLUMN currency TEXT DEFAULT NULL;")

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_created 
            ON transactions(created_at DESC, id DESC);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_fraud 
            ON transactions(is_fraud);
            """
        )
        conn.commit()


def save_transaction(db_path: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a new analyzed transaction into SQLite.
    Preserves the exact supplied transaction ID and updates existing records on conflict.
    """
    txn_id = str(record.get("transactionId") or record.get("transaction_id") or "").strip()
    if not txn_id:
        txn_id = f"TXN-{int(datetime.now(timezone.utc).timestamp() * 1000)}"

    amount = float(record["amount"])
    channel = str(record.get("channel") or "Card")
    merchant_category = str(record.get("merchantCategory") or record.get("merchant_category") or "Other")
    country = str(record.get("country") or "Other")
    hours_since_prev_txn = float(record.get("hoursSincePrevTxn") or record.get("hours_since_prev_txn") or 0.0)
    prediction = str(record.get("prediction") or "Legitimate")
    label = str(record.get("label") or prediction)
    is_fraud = int(record.get("is_fraud", 1 if label == "Fraudulent" else 0))
    fraud_prob = float(record.get("fraud_probability", record.get("fraudProbability", 0.0)))
    confidence = float(record.get("confidence", 1.0 - fraud_prob if is_fraud == 0 else fraud_prob))
    risk_level = str(record.get("risk_level") or record.get("riskLevel") or "Low")
    has_txn_time = record.get("hasTransactionTime")
    if has_txn_time is None:
        has_txn_time = record.get("has_transaction_time")

    provided_ts = (
        record.get("transactionTime")
        or record.get("transaction_time")
        or record.get("timestamp")
    )
    if has_txn_time is None:
        has_txn_time = bool(provided_ts and str(provided_ts).strip())

    timestamp = str(provided_ts or datetime.now(IST_TZ).isoformat())
    created_at = str(
        record.get("created_at")
        or record.get("createdAt")
        or datetime.now(IST_TZ).isoformat()
    )

    raw_currency = record.get("currency")
    if raw_currency is not None and str(raw_currency).strip():
        currency = str(raw_currency).strip().upper()
    else:
        currency = None

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transactions (
                transaction_id, amount, channel, merchant_category, country,
                hours_since_prev_txn, prediction, label, is_fraud,
                fraud_probability, confidence, risk_level, timestamp, created_at,
                currency
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transaction_id) DO UPDATE SET
                amount = excluded.amount,
                channel = excluded.channel,
                merchant_category = excluded.merchant_category,
                country = excluded.country,
                hours_since_prev_txn = excluded.hours_since_prev_txn,
                prediction = excluded.prediction,
                label = excluded.label,
                is_fraud = excluded.is_fraud,
                fraud_probability = excluded.fraud_probability,
                confidence = excluded.confidence,
                risk_level = excluded.risk_level,
                timestamp = excluded.timestamp,
                created_at = excluded.created_at,
                currency = excluded.currency;
            """,
            (
                txn_id,
                amount,
                channel,
                merchant_category,
                country,
                hours_since_prev_txn,
                prediction,
                label,
                is_fraud,
                fraud_prob,
                confidence,
                risk_level,
                timestamp,
                created_at,
                currency,
            ),
        )
        conn.commit()

    return {
        "transactionId": txn_id,
        "transaction_id": txn_id,
        "amount": amount,
        "currency": currency,
        "channel": channel,
        "merchantCategory": merchant_category,
        "merchant_category": merchant_category,
        "country": country,
        "hoursSincePrevTxn": hours_since_prev_txn,
        "hours_since_prev_txn": hours_since_prev_txn,
        "prediction": prediction,
        "label": label,
        "is_fraud": is_fraud,
        "isFraud": is_fraud,
        "fraud_probability": round(fraud_prob, 4),
        "fraudProbability": round(fraud_prob, 4),
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "riskLevel": risk_level,
        "timestamp": timestamp,
        "transactionTime": timestamp if has_txn_time else None,
        "transaction_time": timestamp if has_txn_time else None,
        "hasTransactionTime": bool(has_txn_time),
        "has_transaction_time": bool(has_txn_time),
        "created_at": created_at,
        "createdAt": created_at,
    }


def get_transactions(db_path: str, limit: int = 100) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch transactions sorted newest first, along with total count.
    Uses created_at DESC with id DESC as a deterministic secondary sort key.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM transactions")
        total = cursor.fetchone()["total"]

        cursor.execute(
            """
            SELECT * FROM transactions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        ts = r["timestamp"]
        ca = r["created_at"]
        has_txn_time = False
        if ts and ca:
            try:
                import pandas as _pd
                t_dt = _pd.to_datetime(ts)
                c_dt = _pd.to_datetime(ca)
                if not _pd.isna(t_dt) and not _pd.isna(c_dt):
                    t_epoch = (
                        t_dt.timestamp()
                        if t_dt.tzinfo is not None
                        else t_dt.tz_localize(IST_TZ).timestamp()
                    )
                    c_epoch = (
                        c_dt.timestamp()
                        if c_dt.tzinfo is not None
                        else c_dt.tz_localize(timezone.utc).timestamp()
                    )
                    has_txn_time = abs(t_epoch - c_epoch) > 60
            except Exception:
                has_txn_time = ts != ca

        curr = None
        if "currency" in r.keys() and r["currency"]:
            curr = str(r["currency"]).strip().upper()

        results.append({
            "transactionId": r["transaction_id"],
            "transaction_id": r["transaction_id"],
            "id": r["transaction_id"],
            "amount": float(r["amount"]),
            "currency": curr,
            "channel": r["channel"],
            "merchantCategory": r["merchant_category"],
            "merchant_category": r["merchant_category"],
            "country": r["country"],
            "hoursSincePrevTxn": float(r["hours_since_prev_txn"]),
            "hours_since_prev_txn": float(r["hours_since_prev_txn"]),
            "prediction": r["prediction"],
            "label": r["label"],
            "is_fraud": int(r["is_fraud"]),
            "isFraud": int(r["is_fraud"]),
            "fraud_probability": round(float(r["fraud_probability"]), 4),
            "fraudProbability": round(float(r["fraud_probability"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "risk_level": r["risk_level"],
            "riskLevel": r["risk_level"],
            "timestamp": ts,
            "transactionTime": ts if has_txn_time else None,
            "transaction_time": ts if has_txn_time else None,
            "hasTransactionTime": has_txn_time,
            "has_transaction_time": has_txn_time,
            "created_at": ca,
            "createdAt": ca,
        })

    return results, total


def get_stats(db_path: str) -> Dict[str, Any]:
    """
    Calculate summary transaction statistics:
    - totalTransactions
    - legitimateTransactions
    - fraudulentTransactions
    - fraudRate (percentage 0..100)
    Handles zero transactions safely without division-by-zero errors.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
                SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_count
            FROM transactions
            """
        )
        row = cursor.fetchone()

    total = row["total"] or 0
    fraud_count = row["fraud_count"] or 0
    legit_count = row["legit_count"] or 0

    fraud_rate = round((fraud_count / total * 100.0), 2) if total > 0 else 0.0

    return {
        "totalTransactions": total,
        "legitimateTransactions": legit_count,
        "fraudulentTransactions": fraud_count,
        "fraudRate": fraud_rate,
        # snake_case aliases for backwards compatibility
        "total_transactions": total,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
    }


def get_analytics(db_path: str) -> Dict[str, Any]:
    """
    Aggregate charts data matching frontend schemas:
    - channelBreakdown
    - labelDistribution
    - transactionTrend
    - amountBands
    """
    standard_channels = ["UPI", "Card", "Transfer", "Wallet", "ATM"]
    channel_counts: Dict[str, Dict[str, int]] = {
        ch: {"channel": ch, "legitimate": 0, "flagged": 0} for ch in standard_channels
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # 1. Channel breakdown
        cursor.execute(
            """
            SELECT channel, is_fraud, COUNT(*) AS cnt
            FROM transactions
            GROUP BY channel, is_fraud
            """
        )
        for row in cursor.fetchall():
            ch = row["channel"]
            if ch not in channel_counts:
                channel_counts[ch] = {"channel": ch, "legitimate": 0, "flagged": 0}
            if row["is_fraud"] == 1:
                channel_counts[ch]["flagged"] += row["cnt"]
            else:
                channel_counts[ch]["legitimate"] += row["cnt"]

        # 2. Label distribution
        cursor.execute(
            """
            SELECT 
                SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_count,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count
            FROM transactions
            """
        )
        dist_row = cursor.fetchone()
        legit_total = dist_row["legit_count"] or 0
        fraud_total = dist_row["fraud_count"] or 0

        # 3. Transaction trend (grouped by calendar day)
        cursor.execute(
            """
            SELECT 
                SUBSTR(timestamp, 1, 10) AS day,
                COUNT(*) AS tx_count,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count
            FROM transactions
            GROUP BY SUBSTR(timestamp, 1, 10)
            ORDER BY day ASC
            LIMIT 30
            """
        )
        trend_rows = cursor.fetchall()
        trend_data = [
            {
                "day": r["day"],
                "transactions": r["tx_count"],
                "flagged": r["fraud_count"] or 0,
            }
            for r in trend_rows
        ]

        # 4. Amount bands: Segregate USD amounts from legacy unspecified amounts
        cursor.execute(
            """
            SELECT 
                SUM(CASE WHEN amount < 1000 THEN 1 ELSE 0 END) AS band1,
                SUM(CASE WHEN amount >= 1000 AND amount < 3000 THEN 1 ELSE 0 END) AS band2,
                SUM(CASE WHEN amount >= 3000 AND amount < 6000 THEN 1 ELSE 0 END) AS band3,
                SUM(CASE WHEN amount >= 6000 THEN 1 ELSE 0 END) AS band4,
                COUNT(*) AS total_usd
            FROM transactions
            WHERE currency = 'USD'
            """
        )
        usd_bands_row = cursor.fetchone()
        usd_total_in_bands = usd_bands_row["total_usd"] or 0

        cursor.execute(
            """
            SELECT 
                SUM(CASE WHEN amount < 1000 THEN 1 ELSE 0 END) AS band1,
                SUM(CASE WHEN amount >= 1000 AND amount < 3000 THEN 1 ELSE 0 END) AS band2,
                SUM(CASE WHEN amount >= 3000 AND amount < 6000 THEN 1 ELSE 0 END) AS band3,
                SUM(CASE WHEN amount >= 6000 THEN 1 ELSE 0 END) AS band4,
                COUNT(*) AS total_unspecified
            FROM transactions
            WHERE currency IS NULL OR currency != 'USD'
            """
        )
        unspecified_bands_row = cursor.fetchone()
        unspecified_total_in_bands = unspecified_bands_row["total_unspecified"] or 0

    amount_bands = [
        {"band": "0–1k", "count": usd_bands_row["band1"] or 0},
        {"band": "1k–3k", "count": usd_bands_row["band2"] or 0},
        {"band": "3k–6k", "count": usd_bands_row["band3"] or 0},
        {"band": "6k+", "count": usd_bands_row["band4"] or 0},
    ]

    unspecified_amount_bands = [
        {"band": "0–1k", "count": unspecified_bands_row["band1"] or 0},
        {"band": "1k–3k", "count": unspecified_bands_row["band2"] or 0},
        {"band": "3k–6k", "count": unspecified_bands_row["band3"] or 0},
        {"band": "6k+", "count": unspecified_bands_row["band4"] or 0},
    ]

    label_distribution = [
        {"name": "Legitimate", "value": legit_total},
        {"name": "Fraudulent", "value": fraud_total},
    ]

    return {
        "channelBreakdown": list(channel_counts.values()),
        "labelDistribution": label_distribution,
        "transactionTrend": trend_data,
        "amountBands": amount_bands,
        "amountBandsCurrency": "USD",
        "amountBandsTotal": usd_total_in_bands,
        "unspecifiedAmountBands": unspecified_amount_bands,
        "unspecifiedAmountTotal": unspecified_total_in_bands,
        "amountBandsDisclosure": (
            "Amount distribution aggregates transactions denominated in US Dollars (USD). "
            "Historical records created prior to currency standardization have unspecified "
            "currency units and are segregated to prevent cross-currency distortion."
        ),
        "totalTransactions": legit_total + fraud_total,
        "fraudulentTransactions": fraud_total,
        "legitimateTransactions": legit_total,
    }
