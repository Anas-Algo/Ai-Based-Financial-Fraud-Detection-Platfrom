"""
Automated unit and integration test suite for FraudLens Flask API.
Runs against the Flask test client without needing a live network port.
"""

import json
import os
import shutil
import tempfile
import unittest

from app import DB_PATH, app
from database import IST_TZ, init_db, save_transaction


class FraudLensApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.test_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.test_dir, "test_transactions.db")
        app.config["DB_PATH"] = cls.test_db_path
        init_db(cls.test_db_path)
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        app.config["DB_PATH"] = DB_PATH
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_health_check(self):
        """Verify /api/health returns healthy status and model_loaded=True."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("database"), "connected")

    def test_02_predict_missing_body(self):
        """Verify 400 when body is missing or not JSON."""
        response = self.client.post("/api/predict", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_03_predict_missing_amount(self):
        """Verify 400 when amount is missing."""
        response = self.client.post(
            "/api/predict",
            json={"transactionId": "TXN-TEST-01", "merchantCategory": "Groceries"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("amount is required", response.get_json()["error"].lower())

    def test_04_predict_invalid_amount_strings(self):
        """Verify 400 when amount contains alphabetic characters."""
        response = self.client.post(
            "/api/predict",
            json={"transactionId": "TXN-TEST-02", "amount": "abc250"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid numeric", response.get_json()["error"].lower())

    def test_05_predict_zero_and_negative_amount(self):
        """Verify 400 when amount <= 0."""
        # Zero
        res_zero = self.client.post("/api/predict", json={"amount": 0})
        self.assertEqual(res_zero.status_code, 400)
        self.assertIn("greater than zero", res_zero.get_json()["error"].lower())

        # Negative
        res_neg = self.client.post("/api/predict", json={"amount": -50.0})
        self.assertEqual(res_neg.status_code, 400)
        self.assertIn("greater than zero", res_neg.get_json()["error"].lower())

    def test_06_predict_valid_legitimate_transaction(self):
        """Verify valid transaction prediction and normalization."""
        payload = {
            "transactionId": "TXN-UNIT-LEGIT-01",
            "amount": 45.50,
            "channel": "Card",
            "merchantCategory": "Groceries",
            "country": "India",
            "hoursSincePrevTxn": 12.0,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["transactionId"], "TXN-UNIT-LEGIT-01")
        self.assertEqual(data["prediction"], "Legitimate")
        self.assertEqual(data["label"], "Legitimate")
        self.assertEqual(data["is_fraud"], 0)
        self.assertIsInstance(data["amount"], float)

        # Check bounds: 0 <= fraud_probability <= 1
        fraud_prob = data["fraud_probability"]
        self.assertGreaterEqual(fraud_prob, 0.0)
        self.assertLessEqual(fraud_prob, 1.0)

        # Check bounds: 0 <= confidence <= 1
        conf = data["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIn(data["risk_level"], ["Low", "Medium", "High"])

    def test_07_predict_valid_fraud_transaction(self):
        """Verify prediction for a suspicious high-risk pattern."""
        payload = {
            "transactionId": "TXN-UNIT-FRAUD-01",
            "amount": 281.06,
            "channel": "Card",
            "merchantCategory": "grocery_pos",
            "gender": "M",
            "lat": 35.9946,
            "long": -81.7266,
            "city_pop": 885,
            "merch_lat": 36.430124,
            "merch_long": -81.179483,
            "trans_date_trans_time": "2019-01-02 01:06:37",
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["transactionId"], "TXN-UNIT-FRAUD-01")
        self.assertEqual(data["prediction"], "Fraudulent")
        self.assertEqual(data["label"], "Fraudulent")
        self.assertEqual(data["is_fraud"], 1)
        self.assertGreaterEqual(data["fraud_probability"], 0.70)
        self.assertEqual(data["risk_level"], "High")

    def test_08_transaction_history(self):
        """Verify /api/transactions returns expected shape and newest items first."""
        response = self.client.get("/api/transactions")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("transactions", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["transactions"], list)
        self.assertGreaterEqual(data["total"], 2)

        # Verify newest first (the last inserted fraud transaction should be first)
        first_txn = data["transactions"][0]
        self.assertEqual(first_txn["transactionId"], "TXN-UNIT-FRAUD-01")

    def test_09_stats_endpoint(self):
        """Verify /api/stats calculates metrics accurately without zero-division."""
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("totalTransactions", data)
        self.assertIn("legitimateTransactions", data)
        self.assertIn("fraudulentTransactions", data)
        self.assertIn("fraudRate", data)

        total = data["totalTransactions"]
        legit = data["legitimateTransactions"]
        fraud = data["fraudulentTransactions"]

        self.assertEqual(total, legit + fraud)
        self.assertGreaterEqual(data["fraudRate"], 0.0)
        self.assertLessEqual(data["fraudRate"], 100.0)

    def test_10_analytics_endpoint(self):
        """Verify /api/analytics returns all required chart structures."""
        response = self.client.get("/api/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("channelBreakdown", data)
        self.assertIn("labelDistribution", data)
        self.assertIn("transactionTrend", data)
        self.assertIn("amountBands", data)

        # Check channelBreakdown structure
        for item in data["channelBreakdown"]:
            self.assertIn("channel", item)
            self.assertIn("legitimate", item)
            self.assertIn("flagged", item)

        # Check amountBands
        bands = [b["band"] for b in data["amountBands"]]
        self.assertEqual(bands, ["0–1k", "1k–3k", "3k–6k", "6k+"])

    def test_11_save_transaction_missing_timestamp_uses_ist(self):
        """Verify save_transaction fallback timestamp uses IST (+05:30) timezone."""
        record = {
            "transactionId": "TXN-TZ-TEST-01",
            "amount": 100.0,
            "channel": "UPI",
            "merchantCategory": "Shopping",
        }
        saved = save_transaction(self.test_db_path, record)
        self.assertIn("timestamp", saved)
        # Should contain +05:30 indicating India Standard Time
        self.assertTrue(saved["timestamp"].endswith("+05:30") or "+05:30" in saved["timestamp"])

    def test_12_timestamp_preservation(self):
        """Verify explicit timestamps (with offset, UTC, or offsetless) are strictly preserved."""
        test_timestamps = [
            ("TXN-TZ-PRESERVE-01", "2026-10-23T16:20:28+05:30"),
            ("TXN-TZ-PRESERVE-02", "2026-10-23T10:50:28Z"),
            ("TXN-TZ-PRESERVE-03", "2019-01-02 01:06:37"),
        ]
        for txn_id, raw_ts in test_timestamps:
            record = {
                "transactionId": txn_id,
                "amount": 250.0,
                "channel": "Card",
                "merchantCategory": "Entertainment",
                "timestamp": raw_ts,
            }
            saved = save_transaction(self.test_db_path, record)
            self.assertEqual(saved["timestamp"], raw_ts)


if __name__ == "__main__":
    unittest.main()
