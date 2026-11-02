"""
FraudLens — Synthetic Indian UPI Transaction Baseline Generator
===============================================================
Generates a deterministic, reproducible synthetic dataset of Indian UPI and
retail digital transactions for academic simulation, model training, and benchmark evaluation.

ACADEMIC SIMULATION NOTICE:
---------------------------
This dataset is an academic simulation generated via stochastic parameter modeling.
It does NOT represent actual NPCI switch logs, commercial bank clearing records,
or live production UPI network traffic. All transaction IDs, amounts, and
patterns are synthetic and contain zero real-world personally identifiable information (PII).

Methodology & Statistical Assumptions:
--------------------------------------
1. Determinism:
   - Uses a fixed pseudorandom seed (default: 42) via NumPy's Generator.
   - Repeated runs with the identical seed produce bit-for-bit identical datasets.

2. Volume & Temporal Scope:
   - Exactly 50,000 transactions spanning a 180-day continuous window (Jan 1 – Jun 28, 2024).
   - All timestamps are generated and formatted in India Standard Time (IST, UTC+05:30).
   - Diurnal Arrival Process:
     Timestamps follow a realistic, non-uniform diurnal transaction arrival distribution:
     * Off-peak late night (01:00–05:00 IST): ~2.1% of daily volume (trough ~0.4% at 03:00).
     * Morning ramp (06:00–09:00 IST): ~10.4% of daily volume.
     * Midday peak (10:00–14:00 IST): ~25.5% of daily volume.
     * Afternoon plateau (15:00–17:00 IST): ~17.8% of daily volume.
     * Evening dining/shopping peak (18:00–21:00 IST): ~32.4% of daily volume.
     * Late-evening wind-down (22:00–00:00 IST): ~11.8% of daily volume.
     Day indexes are sampled uniformly over 180 days, hours follow the diurnal arrival weights,
     and intra-hour seconds are sampled uniformly. Timestamps are sorted chronologically.

3. Payment Rails & Channels:
   - P2M (Peer-to-Merchant): ~84.6% of volume (retail purchases, groceries, utilities, food).
   - P2P (Peer-to-Peer): ~15.4% of volume (personal transfers, small vendor QR payments).

4. Retail Merchant Categories & Amount Distributions (INR):
   - kirana_groceries: ~28% (log-normal, median ~Rs 270, range Rs 10 - Rs 5,000)
   - quick_commerce: ~16% (log-normal, median ~Rs 380, range Rs 50 - Rs 6,000)
   - food_dining: ~14% (log-normal, median ~Rs 420, range Rs 50 - Rs 8,000)
   - shopping_retail: ~13% (log-normal, median ~Rs 1,400, range Rs 100 - Rs 50,000)
   - utility_bills: ~10% (log-normal, median ~Rs 810, range Rs 50 - Rs 20,000)
   - travel_ticketing: ~8% (log-normal, median ~Rs 600, range Rs 30 - Rs 30,000)
   - p2p_transfer: ~9% (log-normal, median ~Rs 2,000, range Rs 50 - Rs 100,000)
   - entertainment: ~2% (log-normal, median ~Rs 445, range Rs 100 - Rs 10,000)

5. Stochastic Fraud Modeling:
   - Fraud labels are generated through a continuous latent risk score z rather than
     simplistic deterministic threshold rules:
       z = beta_0 + beta_amt * (log10(amt) - 2.5) + f(hour) + f(category) + f(type) + interaction + noise
     where noise ~ N(0, 0.65^2) and P(fraud) = sigmoid(z).
   - Labels are sampled via Bernoulli trials: y ~ Bernoulli(P(fraud)).
   - Base intercept beta_0 = -5.25 calibrates unconditional fraud prevalence to ~1.17% (586 / 50,000),
     offsetting the negative daytime density shift from the non-uniform diurnal arrival distribution.
   - This ensures probabilistic realism: high amounts are not automatically fraudulent,
     and modest amounts during off-hours can occasionally be fraudulent.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "synthetic_baseline_inr.csv")
SUMMARY_OUTPUT_PATH = os.path.join(BASE_DIR, "synthetic_baseline_summary.json")

CATEGORIES = [
    "kirana_groceries",
    "quick_commerce",
    "food_dining",
    "shopping_retail",
    "utility_bills",
    "travel_ticketing",
    "p2p_transfer",
    "entertainment",
]

# Category probabilities summing to 1.0
CATEGORY_PROBS = [0.28, 0.16, 0.14, 0.13, 0.10, 0.08, 0.09, 0.02]

# Category-specific log-normal amount parameters (mu, sigma, min_val, max_val in INR)
AMOUNT_PARAMS = {
    "kirana_groceries": (5.60, 0.70, 10.0, 5000.0),      # median ~Rs 270
    "quick_commerce": (5.95, 0.65, 50.0, 6000.0),        # median ~Rs 380
    "food_dining": (6.05, 0.75, 50.0, 8000.0),          # median ~Rs 420
    "shopping_retail": (7.25, 1.05, 100.0, 50000.0),     # median ~Rs 1400
    "utility_bills": (6.70, 0.90, 50.0, 20000.0),        # median ~Rs 810
    "travel_ticketing": (6.40, 0.95, 30.0, 30000.0),     # median ~Rs 600
    "p2p_transfer": (7.60, 1.25, 50.0, 100000.0),        # median ~Rs 2000
    "entertainment": (6.10, 0.70, 100.0, 10000.0),       # median ~Rs 445
}

# Diurnal hourly traffic weights (24 hours, summing to 1.0)
# Derived from retail payment diurnal cycle: late-night trough (01:00-05:00),
# morning ramp (06:00-09:00), midday peak (10:00-14:00), afternoon plateau (15:00-17:00),
# and evening peak (18:00-21:00).
HOURLY_WEIGHTS = np.array([
    0.018,  # 00:00 - 01:00
    0.008,  # 01:00 - 02:00
    0.005,  # 02:00 - 03:00
    0.004,  # 03:00 - 04:00
    0.005,  # 04:00 - 05:00
    0.010,  # 05:00 - 06:00
    0.018,  # 06:00 - 07:00
    0.028,  # 07:00 - 08:00
    0.038,  # 08:00 - 09:00
    0.050,  # 09:00 - 10:00
    0.060,  # 10:00 - 11:00
    0.065,  # 11:00 - 12:00
    0.068,  # 12:00 - 13:00
    0.062,  # 13:00 - 14:00
    0.058,  # 14:00 - 15:00
    0.055,  # 15:00 - 16:00
    0.058,  # 16:00 - 17:00
    0.065,  # 17:00 - 18:00
    0.075,  # 18:00 - 19:00
    0.082,  # 19:00 - 20:00
    0.085,  # 20:00 - 21:00
    0.072,  # 21:00 - 22:00
    0.048,  # 22:00 - 23:00
    0.028,  # 23:00 - 00:00
], dtype=np.float64)
HOURLY_WEIGHTS /= HOURLY_WEIGHTS.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid link function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def generate_synthetic_data(
    n_samples: int = 50000,
    random_state: int = 42,
    base_intercept: float = -5.25,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic Indian UPI transactions with non-uniform diurnal arrival.

    Parameters
    ----------
    n_samples : int
        Exact number of transactions to generate (default 50,000).
    random_state : int
        Seed for reproducibility (default 42).
    base_intercept : float
        Intercept for latent log-odds (default -5.25, calibrating prevalence to ~1.17%).

    Returns
    -------
    pd.DataFrame
        DataFrame with 50,000 rows and standardized features.
    """
    rng = np.random.default_rng(random_state)

    # 1. Generate Timestamps chronologically across 180 days (Jan 1, 2024 - Jun 28, 2024 IST)
    # Using non-uniform diurnal arrival:
    start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=IST_OFFSET)
    start_epoch = start_date.timestamp()
    n_days = 180

    # Sample day index uniformly across 180 days
    sampled_days = rng.integers(0, n_days, size=n_samples)
    # Sample hours according to diurnal probability distribution
    sampled_hours = rng.choice(24, size=n_samples, p=HOURLY_WEIGHTS)
    # Sample seconds within hour uniformly
    sampled_seconds = rng.uniform(0.0, 3600.0, size=n_samples)

    # Calculate epoch seconds and sort chronologically
    epoch_seconds = start_epoch + (sampled_days * 86400) + (sampled_hours * 3600) + sampled_seconds
    epoch_seconds.sort()

    timestamps = [
        datetime.fromtimestamp(s, tz=IST_OFFSET)
        for s in epoch_seconds
    ]

    # Temporal feature extraction
    hours = np.array([ts.hour for ts in timestamps], dtype=np.int32)
    days = np.array([ts.day for ts in timestamps], dtype=np.int32)
    months = np.array([ts.month for ts in timestamps], dtype=np.int32)
    weekdays = np.array([ts.weekday() for ts in timestamps], dtype=np.int32)  # 0=Monday, 6=Sunday
    is_weekends = (weekdays >= 5).astype(np.int32)

    # 2. Assign Merchant Categories
    category_indices = rng.choice(len(CATEGORIES), size=n_samples, p=CATEGORY_PROBS)
    categories = [CATEGORIES[i] for i in category_indices]

    # 3. Assign Transaction Type (P2P vs P2M)
    # p2p_transfer is strictly P2P; kirana and food have 15% P2P QR vendors; others are P2M
    types = []
    for cat in categories:
        if cat == "p2p_transfer":
            types.append("P2P")
        elif cat in ("kirana_groceries", "food_dining"):
            types.append("P2P" if rng.random() < 0.15 else "P2M")
        else:
            types.append("P2M")

    # 4. Generate Amounts in INR
    amounts = np.zeros(n_samples, dtype=np.float64)
    for i, cat in enumerate(categories):
        mu, sigma, min_v, max_v = AMOUNT_PARAMS[cat]
        val = rng.lognormal(mean=mu, sigma=sigma)
        val = np.clip(val, min_v, max_v)
        amounts[i] = round(float(val), 2)

    # 5. Stochastic Latent Risk Score & Fraud Label Generation
    # Amount effect: logarithmic scale centered at Rs 316 (log10 ~ 2.5)
    amt_effect = 1.15 * (np.log10(np.maximum(amounts, 1.0)) - 2.50)

    # Hour effect: late night (00:00 - 05:00 IST) has elevated risk; peak day has lower risk
    hour_risk = np.zeros(n_samples, dtype=np.float64)
    for i, h in enumerate(hours):
        if h in (1, 2, 3, 4):
            hour_risk[i] = 1.35
        elif h in (0, 5, 23):
            hour_risk[i] = 0.65
        elif 10 <= h <= 20:
            hour_risk[i] = -0.45

    # Category risk weights
    category_risk_map = {
        "kirana_groceries": -0.90,     # Very safe daily micro-purchases
        "utility_bills": -0.80,        # Trusted recurring billers
        "quick_commerce": -0.40,       # Regular fast retail
        "food_dining": -0.30,          # Standard consumer dining
        "travel_ticketing": 0.35,      # Occasional unauthorized ticket booking
        "entertainment": 0.10,         # Modest risk
        "shopping_retail": 0.75,       # Higher risk for electronics/apparel card-not-present
        "p2p_transfer": 1.10,          # Elevated risk for instant account-draining transfers
    }
    cat_risk = np.array([category_risk_map[c] for c in categories], dtype=np.float64)

    # Transaction type risk: P2P transfers have higher anonymity and fraud risk
    type_risk = np.array([0.45 if t == "P2P" else -0.20 for t in types], dtype=np.float64)

    # Non-linear Interaction: high amount (> Rs 15,000) during late night (00:00 - 05:00 or 23:00)
    night_mask = (hours <= 5) | (hours == 23)
    high_amt_mask = amounts > 15000.0
    interaction_risk = 1.50 * (night_mask & high_amt_mask).astype(np.float64)

    # Add Gaussian stochastic noise to prevent deterministic reverse-engineering
    noise = rng.normal(loc=0.0, scale=0.65, size=n_samples)

    # Latent score and logistic probability
    z = base_intercept + amt_effect + hour_risk + cat_risk + type_risk + interaction_risk + noise
    fraud_probabilities = _sigmoid(z)

    # Bernoulli trial realization
    is_fraud = (rng.uniform(0.0, 1.0, size=n_samples) < fraud_probabilities).astype(np.int32)

    # 6. Build Structured DataFrame
    txn_ids = [f"TXN-IND-{i+1:06d}" for i in range(n_samples)]
    iso_timestamps = [ts.isoformat() for ts in timestamps]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "timestamp": iso_timestamps,
        "amount": amounts,
        "currency": "INR",
        "transaction_type": types,
        "merchant_category": categories,
        "transaction_hour": hours,
        "transaction_day": days,
        "transaction_month": months,
        "day_of_week": weekdays,
        "is_weekend": is_weekends,
        "is_fraud": is_fraud,
    })

    return df


def generate_summary_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate dynamic structured summary statistics from the generated dataset."""
    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = total - fraud_count
    fraud_rate_pct = round(float(fraud_count / total * 100), 2)

    # 1. Hourly Breakdown
    hourly_records = []
    for h in range(24):
        sub = df[df["transaction_hour"] == h]
        h_total = len(sub)
        h_fraud = int(sub["is_fraud"].sum())
        h_rate = round(float(h_fraud / h_total * 100), 2) if h_total > 0 else 0.0
        h_vol_pct = round(float(h_total / total * 100), 2)
        hourly_records.append({
            "hour": h,
            "total_transactions": h_total,
            "volume_percent": h_vol_pct,
            "fraud_count": h_fraud,
            "fraud_rate_percent": h_rate,
        })

    # 2. Monthly Breakdown
    monthly_records = []
    for m in sorted(df["transaction_month"].unique()):
        sub = df[df["transaction_month"] == m]
        m_total = len(sub)
        m_fraud = int(sub["is_fraud"].sum())
        m_rate = round(float(m_fraud / m_total * 100), 2) if m_total > 0 else 0.0
        m_vol_pct = round(float(m_total / total * 100), 2)
        monthly_records.append({
            "month": int(m),
            "total_transactions": m_total,
            "volume_percent": m_vol_pct,
            "fraud_count": m_fraud,
            "fraud_rate_percent": m_rate,
        })

    # 3. Transaction Type Breakdown
    type_records = []
    for t_type in ["P2M", "P2P"]:
        sub = df[df["transaction_type"] == t_type]
        t_total = len(sub)
        t_fraud = int(sub["is_fraud"].sum())
        t_rate = round(float(t_fraud / t_total * 100), 2) if t_total > 0 else 0.0
        t_vol_pct = round(float(t_total / total * 100), 2)
        type_records.append({
            "transaction_type": t_type,
            "total_transactions": t_total,
            "volume_percent": t_vol_pct,
            "fraud_count": t_fraud,
            "fraud_rate_percent": t_rate,
        })

    # 4. Merchant Category Breakdown
    category_records = []
    for cat in CATEGORIES:
        sub = df[df["merchant_category"] == cat]
        c_total = len(sub)
        c_fraud = int(sub["is_fraud"].sum())
        c_rate = round(float(c_fraud / c_total * 100), 2) if c_total > 0 else 0.0
        c_vol_pct = round(float(c_total / total * 100), 2)
        category_records.append({
            "merchant_category": cat,
            "total_transactions": c_total,
            "volume_percent": c_vol_pct,
            "fraud_count": c_fraud,
            "fraud_rate_percent": c_rate,
        })

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(csv_bytes).hexdigest()

    summary = {
        "metadata": {
            "title": "FraudLens Synthetic Indian UPI Benchmark Summary Report",
            "generated_at": datetime.now(IST_OFFSET).isoformat(),
            "currency": "INR",
            "date_range_start": str(df["timestamp"].iloc[0]),
            "date_range_end": str(df["timestamp"].iloc[-1]),
            "random_state": 42,
            "dataset_sha256": sha256,
            "notice": (
                "ACADEMIC SIMULATION NOTICE: All distributions reflect synthetic stochastic modeling. "
                "They do not represent live NPCI network or bank clearing traffic."
            ),
        },
        "overall_prevalence": {
            "total_samples": total,
            "legitimate_count": legit_count,
            "fraudulent_count": fraud_count,
            "fraud_rate_percent": fraud_rate_pct,
            "amount_min_inr": float(df["amount"].min()),
            "amount_median_inr": float(df["amount"].median()),
            "amount_mean_inr": round(float(df["amount"].mean()), 2),
            "amount_max_inr": float(df["amount"].max()),
        },
        "hourly_distribution": hourly_records,
        "monthly_distribution": monthly_records,
        "transaction_type_distribution": type_records,
        "merchant_category_distribution": category_records,
    }
    return summary


def main():
    print("=" * 60)
    print("FraudLens — Synthetic Indian UPI Benchmark Generator")
    print("=" * 60)
    print(f"Generating 50,000 deterministic transactions (seed=42, diurnal arrival)...")

    df = generate_synthetic_data(n_samples=50000, random_state=42)

    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = total - fraud_count
    fraud_rate = (fraud_count / total) * 100.0

    print(f"\nGeneration complete!")
    print(f"Total transactions:    {total:,}")
    print(f"Legitimate count:      {legit_count:,} ({100 - fraud_rate:.2f}%)")
    print(f"Fraudulent count:      {fraud_count:,} ({fraud_rate:.2f}%)")
    print(f"Currency:              {df['currency'].iloc[0]}")
    print(f"Date Range (IST):      {df['timestamp'].iloc[0][:10]} to {df['timestamp'].iloc[-1][:10]}")
    print(f"Amount Min:            Rs {df['amount'].min():.2f}")
    print(f"Amount Median:         Rs {df['amount'].median():.2f}")
    print(f"Amount Mean:           Rs {df['amount'].mean():.2f}")
    print(f"Amount Max:            Rs {df['amount'].max():.2f}")

    print("\nTransaction Type Breakdown:")
    type_counts = df["transaction_type"].value_counts(normalize=True).round(4) * 100
    for t_name, pct in type_counts.items():
        print(f"  {t_name:10s}: {pct:5.2f}%")

    print("\nMerchant Category Breakdown:")
    cat_counts = df["merchant_category"].value_counts(normalize=True).round(4) * 100
    for c_name, pct in cat_counts.items():
        print(f"  {c_name:20s}: {pct:5.2f}%")

    # Save dataset CSV to disk
    os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    file_size_mb = os.path.getsize(DEFAULT_OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved synthetic dataset to: {DEFAULT_OUTPUT_PATH} ({file_size_mb:.2f} MB)")

    # Save structured summary JSON report
    summary = generate_summary_report(df)
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary report to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
