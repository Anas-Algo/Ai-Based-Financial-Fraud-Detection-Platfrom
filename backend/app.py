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

# Allow local frontend development origins (Vite on any port: 3000, 5173, 8080, 8081, etc.)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                re.compile(r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
            ],
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
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_detection_pipeline.pkl")
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
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as exc:
    print(f"Error loading model: {exc}")
    model = None


# ==========================================
# 3. FEATURE SCHEMA & MAPPING
# ==========================================

# Explicit mapping between human-readable frontend categories and dataset categories
CATEGORY_MAPPING = {
    # UI Categories
    "groceries": "grocery_pos",
    "restaurants": "food_dining",
    "electronics": "shopping_pos",
    "clothing": "shopping_pos",
    "travel": "travel",
    "fuel": "gas_transport",
    "healthcare": "health_fitness",
    "entertainment": "entertainment",
    "utilities": "misc_pos",
    "other": "misc_net",
    # Direct dataset category mappings
    "entertainment": "entertainment",
    "food_dining": "food_dining",
    "gas_transport": "gas_transport",
    "grocery_net": "grocery_net",
    "grocery_pos": "grocery_pos",
    "health_fitness": "health_fitness",
    "home": "home",
    "kids_pets": "kids_pets",
    "misc_net": "misc_net",
    "misc_pos": "misc_pos",
    "personal_care": "personal_care",
    "shopping_net": "shopping_net",
    "shopping_pos": "shopping_pos",
}

# The exact feature columns and order expected by the Scikit-learn Pipeline
MODEL_COLUMNS = [
    "category",
    "gender",
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
]


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
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "connected",
        "model_name": "RandomForestClassifier Pipeline",
        "total_records": stats["totalTransactions"],
    })


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Machine learning prediction service is currently unavailable. Model not loaded."
        }), 503

    # 1. Parse JSON body safely
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide transaction details as a valid JSON object."
        }), 400

    # 2. Validate Amount
    # Accept amount or amt
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

    # 3. Transaction ID
    raw_id = str(data.get("transactionId") or data.get("transaction_id") or "").strip()
    if not raw_id:
        raw_id = f"TXN-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    elif len(raw_id) > 100:
        return jsonify({"error": "Transaction ID is too long (maximum 100 characters)."}), 400

    # 4. Merchant Category Mapping
    category_input = str(
        data.get("merchantCategory")
        or data.get("merchant_category")
        or data.get("category")
        or "Other"
    ).strip()

    normalized_cat_key = category_input.lower().replace(" ", "_")
    model_category = CATEGORY_MAPPING.get(normalized_cat_key, "misc_net")

    # 5. Presentation / metadata fields (not fed to ML model)
    # The RandomForest pipeline was trained on credit card POS/ecommerce data and does not have a direct UPI feature column.
    # The user's selected channel (UPI, Card, Transfer, etc.) is preserved as audit metadata in SQLite for display.
    channel = str(data.get("channel") or "UPI").strip()
    country = str(data.get("country") or "Other").strip()
    try:
        hours_since_prev = float(data.get("hoursSincePrevTxn", data.get("hours_since_prev_txn", 0.0)))
        if hours_since_prev < 0:
            hours_since_prev = 0.0
    except (ValueError, TypeError):
        hours_since_prev = 0.0

    # 6. Transaction time & derived features
    # Timestamps default to current India Standard Time (Asia/Kolkata) with explicit offset.
    raw_time = data.get("timestamp", data.get("trans_date_trans_time"))
    if raw_time:
        try:
            tx_dt = pd.to_datetime(raw_time)
            if pd.isna(tx_dt):
                tx_dt = datetime.now(IST_TZ)
        except Exception:
            tx_dt = datetime.now(IST_TZ)
    else:
        tx_dt = datetime.now(IST_TZ)

    # 7. Documented defensible defaults for model features not in frontend form
    # The dataset requires demographic and location features.
    # We use documented neutral/median values rather than fabricating user identity:
    gender = str(data.get("gender") or "F").strip().upper()
    if gender not in ("F", "M"):
        gender = "F"

    try:
        lat = float(data.get("lat", 38.5))
        long = float(data.get("long", -90.2))
        city_pop = float(data.get("city_pop", 50000.0))
        merch_lat = float(data.get("merch_lat", lat))
        merch_long = float(data.get("merch_long", long))
    except (ValueError, TypeError):
        lat, long, city_pop, merch_lat, merch_long = 38.5, -90.2, 50000.0, 38.5, -90.2

    # 8. Construct input DataFrame matching exact training feature schema
    feature_dict = {
        "category": str(model_category),
        "gender": str(gender),
        "amt": float(amount),
        "lat": float(lat),
        "long": float(long),
        "city_pop": float(city_pop),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
        "transaction_hour": int(tx_dt.hour),
        "transaction_day": int(tx_dt.day),
        "transaction_month": int(tx_dt.month),
    }
    input_df = pd.DataFrame([feature_dict])[MODEL_COLUMNS]

    # 9. Model Inference
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

    # 10. Persist transaction into SQLite database
    saved_record = save_transaction(
        get_db_path(),
        {
            "transactionId": raw_id,
            "amount": amount,
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
            "timestamp": tx_dt.isoformat() if hasattr(tx_dt, "isoformat") else str(tx_dt),
        },
    )

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
