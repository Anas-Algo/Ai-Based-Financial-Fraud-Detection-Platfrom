"""
FraudLens — AI-Based Financial Fraud Detection REST API
Provides model inference, transaction audit logging, and aggregated metrics.
"""

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from database import (
    IST_TZ,
    get_analytics,
    get_stats,
    get_transactions,
    init_db,
    save_transaction,
)

# ==========================================
# 1. APPLICATION SETUP & CORS
# ==========================================

app = Flask(__name__)

# Configure CORS origins: permits local development by default; production domains configured via CORS_ALLOWED_ORIGINS
cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
if cors_origins_env:
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        re.compile(r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
    ]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "allow_private_network": True,
        }
    },
)


@app.after_request
def add_cors_and_pna_headers(response):
    """Ensure Chromium Private Network Access (PNA) headers are permitted."""
    if request.headers.get("Access-Control-Request-Private-Network"):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "transactions.db")
DB_PATH = DEFAULT_DB_PATH
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraudlens_india_pipeline.pkl")
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_detection_pipeline.pkl")
app.config.setdefault("DB_PATH", DEFAULT_DB_PATH)


def get_db_path() -> str:
    """Return the currently configured SQLite database path."""
    return app.config.get("DB_PATH", DEFAULT_DB_PATH)


# Initialize database
init_db(get_db_path())


# ==========================================
# 2. LOAD MACHINE LEARNING MODEL
# ==========================================

model = None
fraud_class_idx: int = 1
legit_class_idx: int = 0
explainer = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Fraud detection model loaded successfully from: {MODEL_PATH}")

        # Validate classes
        clf = getattr(model, "named_steps", {}).get("classifier", model)
        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            fraud_class_idx = classes.index(1) if 1 in classes else 1
            legit_class_idx = classes.index(0) if 0 in classes else 0
            print(f"Model classes: {classes} (Fraud index: {fraud_class_idx}, Legit index: {legit_class_idx})")

        # Initialize decision-path explanation engine
        try:
            from explanation import DecisionPathExplainer
            explainer = DecisionPathExplainer(model)
            print("Decision path explanation engine initialized successfully.")
        except Exception as exc:
            print(f"Warning: Could not initialize explanation engine: {exc}")
            explainer = None
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as exc:
    print(f"Error loading model: {exc}")
    model = None
    explainer = None


# ==========================================
# 3. FEATURE SCHEMA & MAPPING (INDIAN UPI MODEL)
# ==========================================

# The 8 exact feature columns and order expected by the fitted Indian UPI Pipeline
MODEL_COLUMNS = [
    "merchant_category",
    "transaction_type",
    "amount",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "day_of_week",
    "is_weekend",
]

# Supported merchant categories for the Indian UPI benchmark model
SUPPORTED_CATEGORIES = {
    "kirana_groceries",
    "quick_commerce",
    "food_dining",
    "shopping_retail",
    "utility_bills",
    "travel_ticketing",
    "p2p_transfer",
    "education",
    "entertainment",
}

# Supported transaction types
SUPPORTED_TRANSACTION_TYPES = {"P2P", "P2M"}

# Supported standard channels (audit metadata)
CHANNEL_MAPPING = {
    "upi": "UPI",
    "card": "Card",
    "transfer": "Transfer",
    "wallet": "Wallet",
    "atm": "ATM",
}

# Mapping user-friendly / frontend category representations to the 8 trained categories
INDIAN_CATEGORY_MAPPING = {
    # Direct model categories
    "kirana_groceries": "kirana_groceries",
    "quick_commerce": "quick_commerce",
    "food_dining": "food_dining",
    "shopping_retail": "shopping_retail",
    "utility_bills": "utility_bills",
    "travel_ticketing": "travel_ticketing",
    "p2p_transfer": "p2p_transfer",
    "education": "education",
    "entertainment": "entertainment",
    # UI and user synonyms
    # Education aliases (covering school, college, tuition, courses, books)
    "school fees": "education",
    "school_fees": "education",
    "school": "education",
    "college fees": "education",
    "college_fees": "education",
    "college": "education",
    "university fees": "education",
    "university_fees": "education",
    "university": "education",
    "coaching fees": "education",
    "coaching_fees": "education",
    "coaching": "education",
    "tuition fees": "education",
    "tuition_fees": "education",
    "tuition": "education",
    "online course": "education",
    "online courses": "education",
    "online_courses": "education",
    "course": "education",
    "courses": "education",
    "books & stationery": "education",
    "books and stationery": "education",
    "books_stationery": "education",
    "books": "education",
    "stationery": "education",
    "educational books": "education",
    "edtech": "education",
    "kirana & groceries": "kirana_groceries",
    "kirana": "kirana_groceries",
    "groceries": "kirana_groceries",
    "grocery": "kirana_groceries",
    "grocery_pos": "kirana_groceries",
    "grocery_net": "kirana_groceries",
    "quick commerce": "quick_commerce",
    "instant delivery": "quick_commerce",
    "instant_delivery": "quick_commerce",
    "food & dining": "food_dining",
    "food": "food_dining",
    "dining": "food_dining",
    "restaurants": "food_dining",
    "shopping & retail": "shopping_retail",
    "shopping": "shopping_retail",
    "retail": "shopping_retail",
    "shopping_pos": "shopping_retail",
    "shopping_net": "shopping_retail",
    "electronics": "shopping_retail",
    "clothing": "shopping_retail",
    "bill payments & utilities": "utility_bills",
    "utility bills": "utility_bills",
    "utilities": "utility_bills",
    "bills": "utility_bills",
    "misc_pos": "utility_bills",
    "travel & ticketing": "travel_ticketing",
    "travel": "travel_ticketing",
    "ticketing": "travel_ticketing",
    "gas_transport": "travel_ticketing",
    "transport": "travel_ticketing",
    "fuel": "travel_ticketing",
    "peer-to-peer transfer": "p2p_transfer",
    "peer to peer transfer": "p2p_transfer",
    "p2p transfer": "p2p_transfer",
    "p2p": "p2p_transfer",
    "transfer": "p2p_transfer",
    "movies": "entertainment",
    "gaming": "entertainment",
}


# ==========================================
# 4. ERROR HANDLERS
# ==========================================

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": getattr(e, "description", "Bad Request")}), 400


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "The requested endpoint was not found."}), 404


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({"error": "Method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({"error": "An internal server error occurred. Please try again later."}), 500


# ==========================================
# 5. PUBLIC & HEALTH ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "FraudLens AI Financial Fraud Detection API",
        "status": "running",
        "model_loaded": model is not None,
    })


@app.route("/api/health", methods=["GET"])
def health():
    stats = get_stats(get_db_path())
    is_loaded = model is not None
    return jsonify({
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "database": "connected",
        "model_name": "Indian UPI RandomForest Pipeline",
        "currency": "INR",
        "total_records": stats["totalTransactions"],
    })


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Machine learning prediction service is currently unavailable. Indian UPI model not loaded."
        }), 503

    # 1. Parse JSON body safely
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide transaction details as a valid JSON object."
        }), 400

    # 2. Validate Amount
    raw_amount = data.get("amount", data.get("amt"))
    if raw_amount is None or raw_amount == "":
        return jsonify({"error": "Transaction amount is required."}), 400

    try:
        amount = float(raw_amount)
    except (ValueError, TypeError):
        return jsonify({"error": "Amount must be a valid numeric value."}), 400

    if math.isnan(amount) or math.isinf(amount):
        return jsonify({"error": "Amount must be a finite number."}), 400

    if amount <= 0:
        return jsonify({"error": "Amount must be greater than zero."}), 400

    if amount > 100_000_000:
        return jsonify({"error": "Amount exceeds maximum permitted limit (100,000,000)."}), 400

    # 3. Validate Required Currency ("INR")
    raw_currency = data.get("currency")
    if raw_currency is None or str(raw_currency).strip() == "":
        return jsonify({
            "error": "Transaction currency is required and must be 'INR'. The model is calibrated exclusively for Indian Rupees (INR)."
        }), 400

    cand_currency = str(raw_currency).strip().upper()
    if cand_currency != "INR":
        return jsonify({
            "error": (
                f"The machine learning model is calibrated exclusively for Indian Rupees (INR). "
                f"Unsupported currency: '{raw_currency}'. Please provide transaction amounts in INR."
            )
        }), 400
    currency = "INR"

    # 4. Validate & Normalize Transaction Type ("P2M" or "P2P")
    raw_tx_type = data.get("transactionType", data.get("transaction_type", data.get("type")))
    if raw_tx_type is not None and str(raw_tx_type).strip():
        cand_tx_type = str(raw_tx_type).strip().upper()
        if cand_tx_type not in SUPPORTED_TRANSACTION_TYPES:
            return jsonify({
                "error": f"Invalid transaction type '{raw_tx_type}'. Supported types: 'P2M' (Person-to-Merchant) and 'P2P' (Peer-to-Peer)."
            }), 400
        tx_type = cand_tx_type
    else:
        tx_type = None  # Will default based on category below

    # 5. Validate & Normalize Merchant Category
    category_input = str(
        data.get("merchantCategory")
        or data.get("merchant_category")
        or data.get("category")
        or ""
    ).strip()

    if not category_input:
        return jsonify({
            "error": f"Merchant category is required. Supported categories: {', '.join(sorted(SUPPORTED_CATEGORIES))}."
        }), 400

    normalized_cat_key = category_input.lower().replace("-", " ").strip()
    model_category = (
        INDIAN_CATEGORY_MAPPING.get(normalized_cat_key)
        or INDIAN_CATEGORY_MAPPING.get(normalized_cat_key.replace(" ", "_"))
        or INDIAN_CATEGORY_MAPPING.get(category_input.lower())
    )

    if not model_category or model_category not in SUPPORTED_CATEGORIES:
        return jsonify({
            "error": f"Invalid or unsupported merchant category: '{category_input}'. Supported categories: {', '.join(sorted(SUPPORTED_CATEGORIES))}."
        }), 400

    # Default transaction_type if not explicitly provided
    if tx_type is None:
        tx_type = "P2P" if model_category == "p2p_transfer" else "P2M"

    # 6. Transaction ID
    raw_id = str(data.get("transactionId") or data.get("transaction_id") or "").strip()
    if not raw_id:
        raw_id = f"TXN-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    elif len(raw_id) > 100:
        return jsonify({"error": "Transaction ID is too long (maximum 100 characters)."}), 400

    # 7. Audit metadata fields (stored in SQLite for display, not fed to 8-feature ML pipeline)
    raw_channel = str(data.get("channel") or "UPI").strip()
    channel = CHANNEL_MAPPING.get(raw_channel.lower(), raw_channel)
    country = str(data.get("country") or "India").strip()
    try:
        hours_since_prev = float(data.get("hoursSincePrevTxn", data.get("hours_since_prev_txn", 0.0)))
        if hours_since_prev < 0:
            hours_since_prev = 0.0
    except (ValueError, TypeError):
        hours_since_prev = 0.0

    # 8. Transaction time vs Recorded/Analysis time (IST)
    raw_time = (
        data.get("transactionTime")
        or data.get("transaction_time")
        or data.get("timestamp")
        or data.get("trans_date_trans_time")
    )
    if isinstance(raw_time, str):
        raw_time = raw_time.strip()

    analysis_now = datetime.now(IST_TZ)
    has_transaction_time = False

    if raw_time:
        try:
            parsed_dt = pd.to_datetime(raw_time)
            if not pd.isna(parsed_dt):
                if parsed_dt.tzinfo is None:
                    tx_dt = parsed_dt.tz_localize(IST_TZ)
                else:
                    tx_dt = parsed_dt.tz_convert(IST_TZ)
                has_transaction_time = True
            else:
                tx_dt = analysis_now
        except Exception:
            tx_dt = analysis_now
    else:
        tx_dt = analysis_now

    # 9. Extract temporal features for Indian model in IST
    transaction_hour = int(tx_dt.hour)
    transaction_day = int(tx_dt.day)
    transaction_month = int(tx_dt.month)
    day_of_week = int(tx_dt.weekday())  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0

    # 10. Construct input DataFrame matching exact 8-feature training schema
    feature_dict = {
        "merchant_category": str(model_category),
        "transaction_type": str(tx_type),
        "amount": float(amount),
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }
    input_df = pd.DataFrame([feature_dict])[MODEL_COLUMNS]

    # 11. Model Inference
    try:
        raw_pred = int(model.predict(input_df)[0])

        clf = getattr(model, "named_steps", {}).get("classifier", model)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            fraud_prob = float(probs[fraud_class_idx])
            pred_prob = float(probs[raw_pred])
        else:
            fraud_prob = 1.0 if raw_pred == 1 else 0.0
            pred_prob = 1.0

        # Enforce valid 0..1 probability bounds
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        confidence = max(0.0, min(1.0, pred_prob))

        label = "Fraudulent" if raw_pred == 1 else "Legitimate"
        prediction = label

        if fraud_prob >= 0.70:
            risk_level = "High"
        elif fraud_prob >= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

    except Exception as exc:
        app.logger.exception("Inference failed")
        return jsonify({
            "error": "The machine learning model encountered an error evaluating this transaction."
        }), 500

    # 12. Persist transaction into SQLite database
    tx_timestamp_str = (
        raw_time
        if (has_transaction_time and isinstance(raw_time, str) and raw_time)
        else (tx_dt.isoformat() if hasattr(tx_dt, "isoformat") else str(tx_dt))
    )
    saved_record = save_transaction(
        get_db_path(),
        {
            "transactionId": raw_id,
            "amount": amount,
            "currency": currency,
            "channel": channel,
            "merchantCategory": category_input,
            "country": country,
            "hoursSincePrevTxn": hours_since_prev,
            "prediction": prediction,
            "label": label,
            "is_fraud": raw_pred,
            "fraud_probability": fraud_prob,
            "confidence": confidence,
            "risk_level": risk_level,
            "timestamp": tx_timestamp_str,
            "has_transaction_time": has_transaction_time,
            "created_at": analysis_now.isoformat(),
        },
    )

    # 13. Generate per-transaction decision-path explanations
    explanations = None
    if explainer is not None:
        try:
            explanations = explainer.explain(input_df)
        except Exception as exc:
            app.logger.warning(f"Could not compute prediction explanations: {exc}")
            explanations = None

    saved_record["explanations"] = explanations

    return jsonify(saved_record), 200


# ==========================================
# 7. TRANSACTION HISTORY ENDPOINT
# ==========================================

@app.route("/api/transactions", methods=["GET"])
def transactions_endpoint():
    try:
        limit = min(int(request.args.get("limit", 100)), 500)
    except (ValueError, TypeError):
        limit = 100

    records, total = get_transactions(get_db_path(), limit=limit)
    return jsonify({
        "transactions": records,
        "total": total,
    }), 200


# ==========================================
# 8. STATISTICS ENDPOINT
# ==========================================

@app.route("/api/stats", methods=["GET"])
def stats_endpoint():
    stats = get_stats(get_db_path())
    return jsonify(stats), 200


# ==========================================
# 9. ANALYTICS ENDPOINT
# ==========================================

@app.route("/api/analytics", methods=["GET"])
def analytics_endpoint():
    analytics = get_analytics(get_db_path())
    return jsonify(analytics), 200


# ==========================================
# 10. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("\nStarting FraudLens API server...")
    print("REST API URL: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
