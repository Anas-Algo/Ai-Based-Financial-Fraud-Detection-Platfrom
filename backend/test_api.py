"""
Automated unit and integration test suite for FraudLens Flask API.
Runs against the Flask test client without needing a live network port.
"""

import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from app import DB_PATH, app, explainer, model
from database import IST_TZ, get_transactions, init_db, save_transaction
from explanation import DecisionPathExplainer


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
        """Verify valid transaction prediction and normalization for Indian UPI model."""
        payload = {
            "transactionId": "TXN-UNIT-LEGIT-01",
            "amount": 250.00,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Kirana & Groceries",
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
        self.assertEqual(data["currency"], "INR")
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
        """Verify prediction for a suspicious high-risk Indian UPI pattern."""
        payload = {
            "transactionId": "TXN-UNIT-FRAUD-01",
            "amount": 85000.0,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2P",
            "merchantCategory": "Peer-to-Peer Transfer",
            "country": "India",
            "trans_date_trans_time": "2024-03-15 02:30:00",
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

    def test_13_historical_dataset_timestamp_distinguished_from_analysis_time(self):
        """Verify historical payment timestamp is distinguished from analysis time."""
        payload = {
            "transactionId": "TXN-HISTORICAL-01",
            "amount": 1800.0,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Shopping & Retail",
            "transactionTime": "2024-01-02 01:06:37",
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["transactionId"], "TXN-HISTORICAL-01")
        self.assertEqual(data["timestamp"], "2024-01-02 01:06:37")
        self.assertEqual(data["transactionTime"], "2024-01-02 01:06:37")
        self.assertTrue(data["hasTransactionTime"])
        self.assertIn("createdAt", data)
        # Analysis time is contemporary (not 2024-01-02)
        self.assertNotIn("2024-01-02", data["createdAt"])

        # Check /api/transactions persistence
        hist_res = self.client.get("/api/transactions")
        self.assertEqual(hist_res.status_code, 200)
        txns = hist_res.get_json()["transactions"]
        match = next((t for t in txns if t["transactionId"] == "TXN-HISTORICAL-01"), None)
        self.assertIsNotNone(match)
        self.assertEqual(match["timestamp"], "2024-01-02 01:06:37")
        self.assertTrue(match["hasTransactionTime"])

    def test_14_manually_submitted_transaction_missing_timestamp(self):
        """Verify transaction submitted without payment time is marked as analysis time."""
        payload = {
            "transactionId": "TXN-MANUAL-NO-TIME-01",
            "amount": 950.0,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Kirana & Groceries",
            "country": "India",
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["transactionId"], "TXN-MANUAL-NO-TIME-01")
        self.assertFalse(data["hasTransactionTime"])
        self.assertIsNone(data["transactionTime"])
        self.assertIn("timestamp", data)
        self.assertIn("createdAt", data)

        # Check in history
        hist_res = self.client.get("/api/transactions")
        txns = hist_res.get_json()["transactions"]
        match = next((t for t in txns if t["transactionId"] == "TXN-MANUAL-NO-TIME-01"), None)
        self.assertIsNotNone(match)
        self.assertFalse(match["hasTransactionTime"])

    def test_15_imputation_handles_missing_uncollected_features(self):
        """Verify that standard Indian UPI inputs produce valid finite predictions and response structure."""
        payload = {
            "transactionId": "TXN-IMPUTE-TEST-01",
            "amount": 1250.75,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Food & Dining",
            "country": "India",
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["transactionId"], "TXN-IMPUTE-TEST-01")
        self.assertIn(data["prediction"], ["Legitimate", "Fraudulent"])
        self.assertIn(data["label"], ["Legitimate", "Fraudulent"])
        self.assertIn(data["is_fraud"], [0, 1])

        # Validate probabilities are finite floats in [0, 1]
        fraud_prob = data["fraud_probability"]
        conf = data["confidence"]
        self.assertTrue(math.isfinite(fraud_prob))
        self.assertTrue(math.isfinite(conf))
        self.assertGreaterEqual(fraud_prob, 0.0)
        self.assertLessEqual(fraud_prob, 1.0)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIn(data["risk_level"], ["Low", "Medium", "High"])

        # Backward compatible response fields
        expected_fields = [
            "transactionId",
            "amount",
            "channel",
            "merchantCategory",
            "country",
            "prediction",
            "label",
            "is_fraud",
            "fraud_probability",
            "confidence",
            "risk_level",
            "timestamp",
            "createdAt",
            "hasTransactionTime",
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing required response field: {field}")

    def test_16_predict_p2p_and_p2m_transactions(self):
        """Verify both P2P and P2M transaction types are accepted and processed."""
        # P2P
        res_p2p = self.client.post(
            "/api/predict",
            json={
                "transactionId": "TXN-TYPE-P2P-01",
                "amount": 500.0,
                "currency": "INR",
                "channel": "UPI",
                "transactionType": "P2P",
                "merchantCategory": "Peer-to-Peer Transfer",
            },
        )
        self.assertEqual(res_p2p.status_code, 200)
        data_p2p = res_p2p.get_json()
        self.assertTrue(math.isfinite(data_p2p["fraud_probability"]))

        # P2M
        res_p2m = self.client.post(
            "/api/predict",
            json={
                "transactionId": "TXN-TYPE-P2M-01",
                "amount": 500.0,
                "currency": "INR",
                "channel": "UPI",
                "transactionType": "P2M",
                "merchantCategory": "Quick Commerce",
            },
        )
        self.assertEqual(res_p2m.status_code, 200)
        data_p2m = res_p2m.get_json()
        self.assertTrue(math.isfinite(data_p2m["fraud_probability"]))

    def test_17_pipeline_fitted_imputer_statistics(self):
        """Verify the Indian pipeline's actual SimpleImputer statistics match dataset learned medians and modes."""
        self.assertIsNotNone(model, "Model pipeline should be loaded")
        preprocessor = model.named_steps["preprocessor"]

        # 1. Numerical imputer verification
        num_imputer = preprocessor.named_transformers_["numerical"].named_steps["imputer"]
        stats = num_imputer.statistics_
        # Expected features: ['amount', 'transaction_hour', 'transaction_day', 'transaction_month', 'day_of_week', 'is_weekend']
        self.assertAlmostEqual(float(stats[0]), 513.10, places=1)  # amount median
        self.assertEqual(float(stats[1]), 15.0)                    # transaction_hour median
        self.assertEqual(float(stats[2]), 15.0)                    # transaction_day median
        self.assertEqual(float(stats[3]), 3.0)                     # transaction_month median
        self.assertEqual(float(stats[4]), 3.0)                     # day_of_week median
        self.assertEqual(float(stats[5]), 0.0)                     # is_weekend median

        # 2. Categorical imputer verification
        cat_imputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]
        cat_stats = cat_imputer.statistics_
        # Expected features: ['merchant_category', 'transaction_type']
        self.assertEqual(str(cat_stats[0]), "kirana_groceries")    # merchant_category mode
        self.assertEqual(str(cat_stats[1]), "P2M")                 # transaction_type mode

        # 3. Transform verification with valid sample
        sample_df = pd.DataFrame([{
            "merchant_category": "kirana_groceries",
            "transaction_type": "P2M",
            "amount": 250.0,
            "transaction_hour": 14,
            "transaction_day": 12,
            "transaction_month": 3,
            "day_of_week": 1,
            "is_weekend": 0,
        }])
        transformed = preprocessor.transform(sample_df)
        # 9 merchant categories + 2 transaction types + 6 numericals = 17 transformed features
        self.assertEqual(transformed.shape[1], 17)
        self.assertTrue(np.all(np.isfinite(transformed if isinstance(transformed, np.ndarray) else transformed.toarray())))

    def test_18_evaluation_report_dual_benchmark_verification(self):
        """Verify that evaluation_report.json contains both full-feature and UI-imputed benchmarks with mathematical consistency."""
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_report.json")
        self.assertTrue(os.path.exists(report_path), f"Report not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Check top-level sections
        self.assertIn("full_feature_evaluation", report)
        self.assertIn("ui_imputed_evaluation", report)
        self.assertIn("comparative_summary", report)
        self.assertIn("dataset_metadata", report)
        self.assertIn("methodology_disclosures", report)

        dataset_meta = report["dataset_metadata"]
        total_eval = dataset_meta["evaluation_samples"]
        fraud_eval = dataset_meta["fraud_samples"]
        legit_eval = dataset_meta["legitimate_samples"]

        self.assertEqual(total_eval, 259335)
        self.assertEqual(fraud_eval, 1501)
        self.assertEqual(legit_eval, 257834)
        self.assertEqual(total_eval, fraud_eval + legit_eval)

        # Verify mathematical consistency for both benchmark configurations
        for config_name in ["full_feature_evaluation", "ui_imputed_evaluation"]:
            sec = report[config_name]
            cm = sec["confusion_matrix"]
            tp = cm["true_positives"]
            fp = cm["false_positives"]
            fn = cm["false_negatives"]
            tn = cm["true_negatives"]

            # 1. Total counts must balance
            self.assertEqual(tp + fp + fn + tn, total_eval, f"{config_name}: CM sum != total_eval")
            self.assertEqual(tp + fn, fraud_eval, f"{config_name}: TP + FN != fraud_eval")
            self.assertEqual(tn + fp, legit_eval, f"{config_name}: TN + FP != legit_eval")

            # 2. Derived metrics check
            calc_prec = tp / (tp + fp)
            calc_rec = tp / (tp + fn)
            calc_f1 = (2 * calc_prec * calc_rec) / (calc_prec + calc_rec)
            calc_acc = (tp + tn) / total_eval

            self.assertAlmostEqual(sec["precision"], calc_prec, places=3, msg=f"{config_name} precision mismatch")
            self.assertAlmostEqual(sec["recall"], calc_rec, places=3, msg=f"{config_name} recall mismatch")
            self.assertAlmostEqual(sec["f1_score"], calc_f1, places=3, msg=f"{config_name} F1 mismatch")
            self.assertAlmostEqual(sec["accuracy"], calc_acc, places=3, msg=f"{config_name} accuracy mismatch")

            # 3. AUC metrics are valid probabilities
            self.assertTrue(0.0 <= sec["average_precision_pr_auc"] <= 1.0)
            self.assertTrue(0.0 <= sec["roc_auc"] <= 1.0)

        # Verify comparative delta
        comp = report["comparative_summary"]
        self.assertEqual(comp["evaluated_samples"], total_eval)
        f_prec = report["full_feature_evaluation"]["precision"]
        i_prec = report["ui_imputed_evaluation"]["precision"]
        self.assertAlmostEqual(comp["metrics_comparison"]["precision"]["delta"], round(i_prec - f_prec, 4), places=4)

        # Verify required disclosures are present
        disclosures = report["methodology_disclosures"]
        self.assertIn("offline_simulation_disclosure", disclosures)
        self.assertIn("masked_inference_features", disclosures)
        self.assertIn("temporal_split_limitation", disclosures)

    def test_19_unseen_cardholder_benchmark_verification(self):
        """Verify that evaluation_report.json contains isolated unseen-cardholder benchmark with disjoint groups and mathematical consistency."""
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_report.json")
        self.assertTrue(os.path.exists(report_path), f"Report not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertIn("unseen_cardholder_benchmark", report, "unseen_cardholder_benchmark missing from report")
        unseen_bench = report["unseen_cardholder_benchmark"]

        # 1. Verify split strategy and disjoint groups
        self.assertIn("split_strategy", unseen_bench)
        split = unseen_bench["split_strategy"]
        self.assertEqual(split["method"], "StratifiedGroupKFold")
        self.assertEqual(split["group_column"], "cc_num")
        self.assertEqual(split["cardholder_overlap_count"], 0)
        self.assertEqual(split["total_unique_cardholders"], 983)
        self.assertEqual(split["train_unique_cardholders"], 786)
        self.assertEqual(split["test_unique_cardholders"], 197)
        self.assertEqual(split["train_unique_cardholders"] + split["test_unique_cardholders"], split["total_unique_cardholders"])

        test_samples = split["test_samples"]
        test_fraud = split["test_fraud_samples"]
        test_legit = split["test_legitimate_samples"]
        self.assertEqual(test_samples, 259242)
        self.assertEqual(test_fraud, 1501)
        self.assertEqual(test_legit, 257741)
        self.assertEqual(test_samples, test_fraud + test_legit)

        # 2. Verify model metadata shows production model protection
        model_meta = unseen_bench["model_metadata"]
        self.assertFalse(model_meta["is_production_model"])
        self.assertTrue(model_meta["production_model_preserved"])

        # 3. Verify mathematical consistency for both unseen-cardholder configurations
        for config_name in ["full_feature_evaluation", "ui_imputed_evaluation"]:
            self.assertIn(config_name, unseen_bench)
            sec = unseen_bench[config_name]
            cm = sec["confusion_matrix"]
            tp = cm["true_positives"]
            fp = cm["false_positives"]
            fn = cm["false_negatives"]
            tn = cm["true_negatives"]

            # Balance checks
            self.assertEqual(tp + fp + fn + tn, test_samples, f"{config_name}: CM sum != test_samples")
            self.assertEqual(tp + fn, test_fraud, f"{config_name}: TP + FN != test_fraud")
            self.assertEqual(tn + fp, test_legit, f"{config_name}: TN + FP != test_legit")

            # Derived metrics checks
            calc_prec = tp / (tp + fp)
            calc_rec = tp / (tp + fn)
            calc_f1 = (2 * calc_prec * calc_rec) / (calc_prec + calc_rec)
            calc_acc = (tp + tn) / test_samples

            self.assertAlmostEqual(sec["precision"], calc_prec, places=3, msg=f"{config_name} precision mismatch")
            self.assertAlmostEqual(sec["recall"], calc_rec, places=3, msg=f"{config_name} recall mismatch")
            self.assertAlmostEqual(sec["f1_score"], calc_f1, places=3, msg=f"{config_name} F1 mismatch")
            self.assertAlmostEqual(sec["accuracy"], calc_acc, places=3, msg=f"{config_name} accuracy mismatch")

            # Probabilistic metrics
            self.assertTrue(0.0 <= sec["average_precision_pr_auc"] <= 1.0)
            self.assertTrue(0.0 <= sec["roc_auc"] <= 1.0)

        # 4. Verify comparative summary
        comp = unseen_bench["comparative_summary"]
        self.assertEqual(comp["evaluated_samples"], test_samples)
        f_prec = unseen_bench["full_feature_evaluation"]["precision"]
        i_prec = unseen_bench["ui_imputed_evaluation"]["precision"]
        self.assertAlmostEqual(comp["metrics_comparison"]["precision"]["delta"], round(i_prec - f_prec, 4), places=4)

        # 5. Verify disclosures
        disclosures = unseen_bench["methodology_disclosures"]
        self.assertIn("offline_simulation_disclosure", disclosures)
        self.assertIn("unseen_cardholder_isolation", disclosures)
        self.assertIn("masked_inference_features", disclosures)

    def test_20_chronological_benchmark_verification(self):
        """Verify that evaluation_report.json contains isolated chronological out-of-time benchmark with strict temporal separation and mathematical consistency."""
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_report.json")
        self.assertTrue(os.path.exists(report_path), f"Report not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # 1. Verify all previous benchmark sections remain present
        self.assertIn("full_feature_evaluation", report)
        self.assertIn("ui_imputed_evaluation", report)
        self.assertIn("comparative_summary", report)
        self.assertIn("unseen_cardholder_benchmark", report)
        self.assertIn("chronological_out_of_time_benchmark", report, "chronological_out_of_time_benchmark missing from report")

        oot_bench = report["chronological_out_of_time_benchmark"]

        # 2. Verify split strategy and temporal separation
        self.assertIn("split_strategy", oot_bench)
        split = oot_bench["split_strategy"]
        self.assertEqual(split["method"], "Chronological Holdout (Out-of-Time)")
        self.assertEqual(split["cutoff_timestamp"], "2020-03-06 07:16:43")

        # Timestamp boundaries check
        cutoff_dt = pd.to_datetime(split["cutoff_timestamp"])
        train_end_dt = pd.to_datetime(split["training_end_timestamp"])
        test_start_dt = pd.to_datetime(split["evaluation_start_timestamp"])
        self.assertLess(train_end_dt, cutoff_dt, "Training data contains transactions on/after cutoff")
        self.assertGreaterEqual(test_start_dt, cutoff_dt, "Evaluation data contains transactions before cutoff")

        # Sample and fraud counts
        train_samples = split["train_samples"]
        train_fraud = split["train_fraud_samples"]
        train_legit = split["train_legitimate_samples"]
        test_samples = split["test_samples"]
        test_fraud = split["test_fraud_samples"]
        test_legit = split["test_legitimate_samples"]

        self.assertEqual(train_samples, 1037340)
        self.assertEqual(train_fraud, 5968)
        self.assertEqual(train_legit, 1031372)
        self.assertEqual(train_samples, train_fraud + train_legit)

        self.assertEqual(test_samples, 259335)
        self.assertEqual(test_fraud, 1538)
        self.assertEqual(test_legit, 257797)
        self.assertEqual(test_samples, test_fraud + test_legit)

        self.assertEqual(train_samples + test_samples, 1296675)
        self.assertEqual(train_fraud + test_fraud, 7506)

        # 3. Verify model metadata shows production model protection and train-only preprocessing
        model_meta = oot_bench["model_metadata"]
        self.assertFalse(model_meta["is_production_model"])
        self.assertTrue(model_meta["production_model_preserved"])
        self.assertTrue(model_meta["preprocessing_fitted_on_train_only"])

        # 4. Verify mathematical consistency for both chronological configurations
        for config_name in ["full_feature_evaluation", "ui_imputed_evaluation"]:
            self.assertIn(config_name, oot_bench)
            sec = oot_bench[config_name]
            cm = sec["confusion_matrix"]
            tp = cm["true_positives"]
            fp = cm["false_positives"]
            fn = cm["false_negatives"]
            tn = cm["true_negatives"]

            # Balance checks
            self.assertEqual(tp + fp + fn + tn, test_samples, f"{config_name}: CM sum != test_samples")
            self.assertEqual(tp + fn, test_fraud, f"{config_name}: TP + FN != test_fraud")
            self.assertEqual(tn + fp, test_legit, f"{config_name}: TN + FP != test_legit")

            # Derived metrics checks
            calc_prec = tp / (tp + fp)
            calc_rec = tp / (tp + fn)
            calc_f1 = (2 * calc_prec * calc_rec) / (calc_prec + calc_rec)
            calc_acc = (tp + tn) / test_samples

            self.assertAlmostEqual(sec["precision"], calc_prec, places=3, msg=f"{config_name} precision mismatch")
            self.assertAlmostEqual(sec["recall"], calc_rec, places=3, msg=f"{config_name} recall mismatch")
            self.assertAlmostEqual(sec["f1_score"], calc_f1, places=3, msg=f"{config_name} F1 mismatch")
            self.assertAlmostEqual(sec["accuracy"], calc_acc, places=3, msg=f"{config_name} accuracy mismatch")

            # Probabilistic metrics
            self.assertTrue(0.0 <= sec["average_precision_pr_auc"] <= 1.0)
            self.assertTrue(0.0 <= sec["roc_auc"] <= 1.0)

        # 5. Verify comparative summary
        comp = oot_bench["comparative_summary"]
        self.assertEqual(comp["evaluated_samples"], test_samples)
        f_prec = oot_bench["full_feature_evaluation"]["precision"]
        i_prec = oot_bench["ui_imputed_evaluation"]["precision"]
        self.assertAlmostEqual(comp["metrics_comparison"]["precision"]["delta"], round(i_prec - f_prec, 4), places=4)

        # 6. Verify disclosures
        disclosures = oot_bench["methodology_disclosures"]
        self.assertIn("offline_simulation_disclosure", disclosures)
        self.assertIn("chronological_out_of_time_separation", disclosures)
        self.assertIn("cardholder_overlap_distinction", disclosures)
        self.assertIn("masked_inference_features", disclosures)

    def test_21_explanation_probability_reconstruction_and_additivity(self):
        """Verify decision-path probability reconstruction and additivity within 1e-7 across diverse Indian UPI inputs."""
        self.assertIsNotNone(explainer, "DecisionPathExplainer must be initialized")

        test_inputs = [
            {"amount": 150.0, "merchant_category": "kirana_groceries", "transaction_type": "P2M", "transaction_hour": 10, "transaction_day": 3, "transaction_month": 5, "day_of_week": 4, "is_weekend": 0},
            {"amount": 1250.50, "merchant_category": "travel_ticketing", "transaction_type": "P2M", "transaction_hour": 8, "transaction_day": 14, "transaction_month": 8, "day_of_week": 2, "is_weekend": 0},
            {"amount": 9800.0, "merchant_category": "shopping_retail", "transaction_type": "P2M", "transaction_hour": 23, "transaction_day": 21, "transaction_month": 11, "day_of_week": 3, "is_weekend": 0},
            {"amount": 35000.0, "merchant_category": "p2p_transfer", "transaction_type": "P2P", "transaction_hour": 2, "transaction_day": 1, "transaction_month": 1, "day_of_week": 6, "is_weekend": 1},
            {"amount": 450.0, "merchant_category": "food_dining", "transaction_type": "P2M", "transaction_hour": 14, "transaction_day": 15, "transaction_month": 6, "day_of_week": 5, "is_weekend": 1},
        ]

        for idx, t_in in enumerate(test_inputs):
            df = pd.DataFrame([t_in])
            res = explainer.explain(df)

            # Reconstruction error check: strictly within 1e-7
            self.assertLess(res["reconstruction_error"], 1e-7, f"Input {idx}: Reconstruction error exceeded 1e-7")

            # Probability bounds
            self.assertTrue(0.0 <= res["base_probability"] <= 1.0)
            self.assertTrue(0.0 <= res["predicted_probability"] <= 1.0)

            # Direct additivity check: base_probability + sum(feat_contribs) == predicted_probability
            model_prob = float(model.predict_proba(df)[0, explainer.fraud_class_idx])
            self.assertAlmostEqual(res["predicted_probability"], model_prob, places=4)

    def test_22_explanation_determinism(self):
        """Verify that repeated explanations for the identical Indian transaction are bit-for-bit deterministic."""
        self.assertIsNotNone(explainer)

        df = pd.DataFrame([{
            "merchant_category": "shopping_retail",
            "transaction_type": "P2M",
            "amount": 8500.0,
            "transaction_hour": 22,
            "transaction_day": 18,
            "transaction_month": 9,
            "day_of_week": 2,
            "is_weekend": 0,
        }])

        first = explainer.explain(df)
        for _ in range(5):
            repeated = explainer.explain(df)
            self.assertEqual(first["base_probability"], repeated["base_probability"])
            self.assertEqual(first["predicted_probability"], repeated["predicted_probability"])
            self.assertEqual(first["imputed_context_contribution"], repeated["imputed_context_contribution"])
            self.assertEqual(first["effective_baseline_probability"], repeated["effective_baseline_probability"])
            self.assertEqual(len(first["factors"]), len(repeated["factors"]))
            for f1, f2 in zip(first["factors"], repeated["factors"]):
                self.assertEqual(f1["feature"], f2["feature"])
                self.assertEqual(f1["value"], f2["value"])
                self.assertEqual(f1["contribution"], f2["contribution"])
                self.assertEqual(f1["impact_percentage"], f2["impact_percentage"])
                self.assertEqual(f1["direction"], f2["direction"])

    def test_23_explanation_no_raw_encoder_dummies_in_user_factors(self):
        """Verify that raw one-hot encoder dummy column names are never presented as factors."""
        self.assertIsNotNone(explainer)

        df = pd.DataFrame([{
            "merchant_category": "shopping_retail",
            "transaction_type": "P2M",
            "amount": 2500.0,
            "transaction_hour": 15,
            "transaction_day": 10,
            "transaction_month": 4,
            "day_of_week": 3,
            "is_weekend": 0,
        }])

        res = explainer.explain(df)
        factor_features = [f["feature"] for f in res["factors"]]

        valid_features = {
            "Amount",
            "Merchant Category",
            "Transaction Type",
            "Transaction Hour",
            "Transaction Day",
            "Transaction Month",
            "Day of Week",
            "Weekend",
            "Weekend Indicator",
        }
        for f_name in factor_features:
            self.assertIn(f_name, valid_features)
            self.assertFalse(f_name.startswith("categorical__"))
            self.assertFalse("onehot" in f_name.lower())

    def test_24_explanation_merchant_category_aggregation(self):
        """Verify that all one-hot category dummy contributions are aggregated into one Merchant Category factor."""
        self.assertIsNotNone(explainer)

        df = pd.DataFrame([{
            "merchant_category": "shopping_retail",
            "transaction_type": "P2M",
            "amount": 9500.0,
            "transaction_hour": 23,
            "transaction_day": 12,
            "transaction_month": 7,
            "day_of_week": 4,
            "is_weekend": 0,
        }])

        res = explainer.explain(df)
        category_factors = [f for f in res["factors"] if f["feature"] == "Merchant Category"]

        # Must have at most one Merchant Category factor
        self.assertLessEqual(len(category_factors), 1)
        if category_factors:
            cat_f = category_factors[0]
            self.assertEqual(cat_f["value"], "shopping_retail")
            self.assertIn("shopping_retail", cat_f["description"])

        # Ensure no raw one-hot dummy column names leak into the explanation
        for f in res["factors"]:
            self.assertFalse(f["feature"].startswith("categorical__"))
            self.assertFalse("category_" in f["feature"])

    def test_25_api_predict_backward_compatibility_with_explanations(self):
        """Verify POST /api/predict response preserves all existing fields and includes valid explanations."""
        payload = {
            "transactionId": "TXN-EXPL-001",
            "amount": 950.0,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Shopping & Retail",
            "country": "India",
            "hoursSincePrevTxn": 6.5,
        }

        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Verify existing contract fields
        expected_fields = [
            "transactionId", "amount", "channel", "merchantCategory", "country",
            "hoursSincePrevTxn", "prediction", "label", "is_fraud",
            "fraud_probability", "confidence", "risk_level", "timestamp", "created_at",
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing expected field: {field}")

        # Verify explanations payload
        self.assertIn("explanations", data)
        exp = data["explanations"]
        self.assertIsNotNone(exp)
        self.assertEqual(exp["attribution_method"], "Decision Tree Path Decomposition (Saabas)")
        self.assertEqual(exp["target_class"], "Fraudulent")
        self.assertIn("base_probability", exp)
        self.assertIn("predicted_probability", exp)
        self.assertIn("effective_baseline_probability", exp)
        self.assertIn("factors", exp)
        self.assertIn("disclaimer", exp)
        self.assertIn("associative patterns", exp["disclaimer"].lower())
        self.assertNotIn("confirmed fraud", exp["disclaimer"].lower())

    def test_26_explanation_edge_cases_and_error_handling(self):
        """Verify explanation engine handling of edge cases and input validation errors."""
        self.assertIsNotNone(explainer)

        # 1. Edge case: extremely small amount (Rs 0.01)
        small_df = pd.DataFrame([{
            "merchant_category": "kirana_groceries",
            "transaction_type": "P2M",
            "amount": 0.01,
            "transaction_hour": 12,
            "transaction_day": 1,
            "transaction_month": 1,
            "day_of_week": 0,
            "is_weekend": 0,
        }])
        small_res = explainer.explain(small_df)
        self.assertLess(small_res["reconstruction_error"], 1e-7)

        # 2. Edge case: maximum allowable amount (Rs 100,000,000)
        large_df = pd.DataFrame([{
            "merchant_category": "shopping_retail",
            "transaction_type": "P2M",
            "amount": 100_000_000.0,
            "transaction_hour": 23,
            "transaction_day": 31,
            "transaction_month": 12,
            "day_of_week": 4,
            "is_weekend": 0,
        }])
        large_res = explainer.explain(large_df)
        self.assertLess(large_res["reconstruction_error"], 1e-7)

        # 3. Invalid inputs to explain(): non-DataFrame or empty DataFrame
        with self.assertRaises(TypeError):
            explainer.explain({"not": "a dataframe"})

        with self.assertRaises(ValueError):
            explainer.explain(pd.DataFrame())  # 0 rows

        with self.assertRaises(ValueError):
            explainer.explain(pd.concat([small_df, large_df]))  # 2 rows

    # ========================================================
    # Currency Standardization & Legacy Preservation Tests
    # ========================================================

    def test_27_database_migration_preserves_records_and_is_idempotent(self):
        """Verify migration adds nullable currency column, preserves existing records, and is idempotent."""
        import sqlite3

        temp_dir = tempfile.mkdtemp()
        old_db_path = os.path.join(temp_dir, "legacy.db")
        try:
            # 1. Create a legacy database without currency column
            conn = sqlite3.connect(old_db_path)
            conn.execute(
                """
                CREATE TABLE transactions (
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
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, amount, channel, merchant_category, country,
                    hours_since_prev_txn, prediction, label, is_fraud,
                    fraud_probability, confidence, risk_level, timestamp, created_at
                ) VALUES ('LEGACY-01', 3150.0, 'UPI', 'Other', 'India', 0.0, 'Legitimate', 'Legitimate', 0, 0.01, 0.99, 'Low', '2026-10-24T00:00:00', '2026-10-24T00:00:00');
                """
            )
            conn.commit()
            conn.close()

            # 2. Run init_db to execute non-destructive migration
            init_db(old_db_path)

            # 3. Verify column was added and existing record preserved with currency=NULL
            conn = sqlite3.connect(old_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(transactions)")
            cols = [r["name"] for r in cursor.fetchall()]
            self.assertIn("currency", cols)

            cursor.execute("SELECT * FROM transactions WHERE transaction_id = 'LEGACY-01'")
            row = cursor.fetchone()
            self.assertEqual(row["amount"], 3150.0)
            self.assertIsNone(row["currency"])
            conn.close()

            # 4. Run init_db again to verify idempotency
            init_db(old_db_path)
            conn = sqlite3.connect(old_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM transactions")
            self.assertEqual(cursor.fetchone()[0], 1)
            conn.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_28_new_inr_transaction_persistence(self):
        """Verify save_transaction explicitly stores currency='INR'."""
        record = {
            "transactionId": "TXN-CURR-INR-01",
            "amount": 450.0,
            "currency": "INR",
            "channel": "UPI",
            "merchantCategory": "Kirana & Groceries",
            "country": "India",
            "prediction": "Legitimate",
            "label": "Legitimate",
            "is_fraud": 0,
            "fraud_probability": 0.01,
            "confidence": 0.99,
            "risk_level": "Low",
        }
        res = save_transaction(self.test_db_path, record)
        self.assertEqual(res["currency"], "INR")

        # Verify from database
        txs, _ = get_transactions(self.test_db_path, limit=50)
        saved = next(t for t in txs if t["transactionId"] == "TXN-CURR-INR-01")
        self.assertEqual(saved["currency"], "INR")

    def test_29_explicit_currency_validation(self):
        """Verify 400 Bad Request when unsupported currency (USD, EUR) is submitted and INR is accepted."""
        # USD currency should be rejected with 400
        res_usd = self.client.post(
            "/api/predict",
            json={"amount": 2500, "currency": "USD", "merchantCategory": "Kirana & Groceries"},
        )
        self.assertEqual(res_usd.status_code, 400)
        self.assertIn("calibrated exclusively for indian rupees", res_usd.get_json()["error"].lower())

        # EUR currency should be rejected with 400
        res_eur = self.client.post(
            "/api/predict",
            json={"amount": 50, "currency": "EUR", "merchantCategory": "Kirana & Groceries"},
        )
        self.assertEqual(res_eur.status_code, 400)
        self.assertIn("unsupported currency", res_eur.get_json()["error"].lower())

        # Valid INR currency accepted
        res_inr = self.client.post(
            "/api/predict",
            json={
                "amount": 500,
                "currency": "INR",
                "transactionType": "P2M",
                "merchantCategory": "Kirana & Groceries",
            },
        )
        self.assertEqual(res_inr.status_code, 200)
        self.assertEqual(res_inr.get_json()["currency"], "INR")

    def test_30_missing_currency_rejected_with_400(self):
        """Verify missing or empty currency in predict payload returns HTTP 400 Bad Request."""
        res_missing = self.client.post(
            "/api/predict",
            json={"amount": 650.0, "merchantCategory": "Kirana & Groceries"},
        )
        self.assertEqual(res_missing.status_code, 400)
        self.assertIn("currency is required", res_missing.get_json()["error"].lower())

        res_empty = self.client.post(
            "/api/predict",
            json={"amount": 650.0, "currency": "   ", "merchantCategory": "Kirana & Groceries"},
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("currency is required", res_empty.get_json()["error"].lower())

    def test_31_get_transactions_currency_serialization(self):
        """Verify GET /api/transactions returns currency as 'INR', 'USD', or None for legacy records."""
        # Save a legacy-like record with currency=None
        save_transaction(
            self.test_db_path,
            {
                "transactionId": "TXN-LEGACY-NULL",
                "amount": 200.0,
                "currency": None,
                "channel": "Card",
                "merchantCategory": "Other",
                "country": "India",
            },
        )
        # Save a USD record
        save_transaction(
            self.test_db_path,
            {
                "transactionId": "TXN-LEGACY-USD",
                "amount": 75.0,
                "currency": "USD",
                "channel": "Card",
                "merchantCategory": "Other",
                "country": "US",
            },
        )
        # Save an INR record
        save_transaction(
            self.test_db_path,
            {
                "transactionId": "TXN-MODERN-INR",
                "amount": 1500.0,
                "currency": "INR",
                "channel": "UPI",
                "merchantCategory": "Kirana & Groceries",
                "country": "India",
            },
        )

        res = self.client.get("/api/transactions?limit=50")
        self.assertEqual(res.status_code, 200)
        txs = res.get_json()["transactions"]

        legacy_item = next(t for t in txs if t["transactionId"] == "TXN-LEGACY-NULL")
        self.assertIsNone(legacy_item["currency"])

        usd_item = next(t for t in txs if t["transactionId"] == "TXN-LEGACY-USD")
        self.assertEqual(usd_item["currency"], "USD")

        inr_item = next(t for t in txs if t["transactionId"] == "TXN-MODERN-INR")
        self.assertEqual(inr_item["currency"], "INR")

    def test_32_analytics_separation_of_inr_amounts(self):
        """Verify GET /api/analytics segregates INR amount bands from USD and unspecified legacy amounts."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "analytics_test.db")
        try:
            init_db(db_path)
            # Insert 1 INR transaction of Rs 450 (band 0-1k)
            save_transaction(
                db_path,
                {"transactionId": "TX-INR-1", "amount": 450.0, "currency": "INR", "is_fraud": 0},
            )
            # Insert 1 USD transaction of $45 (band 0-1k in USD)
            save_transaction(
                db_path,
                {"transactionId": "TX-USD-1", "amount": 45.0, "currency": "USD", "is_fraud": 0},
            )
            # Insert 1 legacy unspecified transaction of 3150 (band 3k-6k)
            save_transaction(
                db_path,
                {"transactionId": "TX-LEG-1", "amount": 3150.0, "currency": None, "is_fraud": 0},
            )

            from database import get_analytics
            analytics = get_analytics(db_path)

            self.assertEqual(analytics["amountBandsCurrency"], "INR")
            self.assertEqual(analytics["amountBandsTotal"], 1)
            # INR band 0-1k should have count 1, all other INR bands 0
            inr_band_0_1k = next(b for b in analytics["amountBands"] if b["band"] == "0–1k")
            self.assertEqual(inr_band_0_1k["count"], 1)
            inr_band_3_6k = next(b for b in analytics["amountBands"] if b["band"] == "3k–6k")
            self.assertEqual(inr_band_3_6k["count"], 0)

            # Segregated non-INR count should be 2 (1 USD + 1 legacy unspecified)
            self.assertEqual(analytics["unspecifiedAmountTotal"], 2)

            # Disclosure is present and mentions Indian Rupees
            self.assertIn("amountBandsDisclosure", analytics)
            self.assertIn("Indian Rupees", analytics["amountBandsDisclosure"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_33_invalid_transaction_type_rejection(self):
        """Verify 400 Bad Request when invalid transaction type is provided."""
        res = self.client.post(
            "/api/predict",
            json={
                "amount": 500,
                "currency": "INR",
                "transactionType": "INVALID_TYPE",
                "merchantCategory": "Kirana & Groceries",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("invalid transaction type", res.get_json()["error"].lower())

    def test_34_invalid_merchant_category_rejection(self):
        """Verify 400 Bad Request when unmapped or invalid merchant category is provided."""
        res = self.client.post(
            "/api/predict",
            json={
                "amount": 500,
                "currency": "INR",
                "transactionType": "P2M",
                "merchantCategory": "Cryptocurrency_Mining",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("invalid or unsupported merchant category", res.get_json()["error"].lower())

    def test_35_education_valid_p2m_transaction(self):
        """Verify valid education P2M transaction prediction and response structure."""
        payload = {
            "transactionId": "TXN-EDU-P2M-01",
            "amount": 4500.0,
            "currency": "INR",
            "channel": "UPI",
            "transactionType": "P2M",
            "merchantCategory": "Education",
            "country": "India",
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["currency"], "INR")
        self.assertIn(data["prediction"], ["Legitimate", "Fraudulent"])
        self.assertIn(data["risk_level"], ["Low", "Medium", "High"])
        self.assertTrue(0.0 <= data["fraud_probability"] <= 1.0)
        self.assertTrue(0.0 <= data["confidence"] <= 1.0)
        self.assertIn("explanations", data)

    def test_36_education_category_normalization(self):
        """Verify that various education aliases normalize to canonical education."""
        aliases = [
            "School Fees",
            "College Fees",
            "Coaching Fees",
            "Tuition",
            "Online Course",
            "Books & Stationery",
            "educational books",
            "edtech",
        ]
        for alias in aliases:
            res = self.client.post(
                "/api/predict",
                json={
                    "transactionId": f"TXN-ALIAS-{alias[:6]}",
                    "amount": 1200.0,
                    "currency": "INR",
                    "channel": "UPI",
                    "transactionType": "P2M",
                    "merchantCategory": alias,
                    "country": "India",
                },
            )
            self.assertEqual(res.status_code, 200, f"Alias '{alias}' failed with status {res.status_code}")
            data = res.get_json()
            self.assertEqual(data["currency"], "INR")

    def test_37_invalid_education_alias_rejection(self):
        """Verify unsupported or invalid education variants return HTTP 400."""
        invalid_aliases = [
            "education_xyz_unknown",
            "rocket_science_academy_secret",
            "flying_school_unsupported",
        ]
        for inv in invalid_aliases:
            res = self.client.post(
                "/api/predict",
                json={
                    "amount": 2000.0,
                    "currency": "INR",
                    "transactionType": "P2M",
                    "merchantCategory": inv,
                },
            )
            self.assertEqual(res.status_code, 400)
            self.assertIn("invalid or unsupported merchant category", res.get_json()["error"].lower())

    def test_38_education_prediction_using_trained_model(self):
        """Verify direct in-memory model inference on education category without warnings."""
        import pandas as pd
        input_df = pd.DataFrame([{
            "merchant_category": "education",
            "transaction_type": "P2M",
            "amount": 5500.0,
            "transaction_hour": 14,
            "transaction_day": 10,
            "transaction_month": 4,
            "day_of_week": 2,
            "is_weekend": 0,
        }])
        pred = model.predict(input_df)
        probs = model.predict_proba(input_df)
        self.assertIn(int(pred[0]), [0, 1])
        self.assertEqual(len(probs[0]), 2)
        self.assertAlmostEqual(float(probs[0].sum()), 1.0, places=5)

    def test_39_education_in_fitted_onehot_encoder_categories(self):
        """Verify that 'education' is present in the fitted OneHotEncoder categories."""
        encoder = model.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
        cat_classes = list(encoder.categories_[0])
        self.assertIn("education", cat_classes)
        self.assertEqual(len(cat_classes), 9, f"Expected 9 categories, got {len(cat_classes)}: {cat_classes}")

    def test_40_existing_categories_still_accepted(self):
        """Verify that all 8 original categories continue to be accepted and evaluated."""
        original_categories = [
            "Kirana & Groceries",
            "Quick Commerce",
            "Food & Dining",
            "Shopping & Retail",
            "Bill Payments & Utilities",
            "Travel & Ticketing",
            "Peer-to-Peer Transfer",
            "Entertainment",
        ]
        for cat in original_categories:
            tx_type = "P2P" if "Peer" in cat else "P2M"
            res = self.client.post(
                "/api/predict",
                json={
                    "amount": 750.0,
                    "currency": "INR",
                    "channel": "UPI",
                    "transactionType": tx_type,
                    "merchantCategory": cat,
                    "country": "India",
                },
            )
            self.assertEqual(res.status_code, 200, f"Category '{cat}' failed with status {res.status_code}")

    def test_41_education_inr_validation_enforced(self):
        """Verify that education transactions still strictly require INR currency."""
        res = self.client.post(
            "/api/predict",
            json={
                "amount": 5000.0,
                "currency": "USD",
                "channel": "Card",
                "transactionType": "P2M",
                "merchantCategory": "Education",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("unsupported currency", res.get_json()["error"].lower())

    def test_42_education_p2p_and_p2m_and_channels(self):
        """Verify that education accepts both P2M/P2P and preserves standard payment channels."""
        channels = ["UPI", "Card", "Transfer", "Wallet", "ATM"]
        for ch in channels:
            for t_type in ["P2M", "P2P"]:
                res = self.client.post(
                    "/api/predict",
                    json={
                        "amount": 2500.0,
                        "currency": "INR",
                        "channel": ch,
                        "transactionType": t_type,
                        "merchantCategory": "Education",
                        "country": "India",
                    },
                )
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertEqual(data["channel"], ch)

    def test_43_education_database_persistence_and_isolation(self):
        """Verify education persistence in test DB and confirm real transactions.db remains isolated."""
        import sqlite3
        res = self.client.post(
            "/api/predict",
            json={
                "transactionId": "TXN-EDU-PERSIST-01",
                "amount": 8000.0,
                "currency": "INR",
                "channel": "UPI",
                "transactionType": "P2M",
                "merchantCategory": "Education",
                "country": "India",
            },
        )
        self.assertEqual(res.status_code, 200)

        # Check test DB
        with sqlite3.connect(self.test_db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT amount, currency, channel FROM transactions WHERE transaction_id = 'TXN-EDU-PERSIST-01'")
            row = c.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 8000.0)
            self.assertEqual(row[1], "INR")
            self.assertEqual(row[2], "UPI")

        # Check real DB isolation (must still have exactly 34 records)
        real_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.db")
        if os.path.exists(real_db):
            with sqlite3.connect(real_db) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM transactions")
                cnt = c.fetchone()[0]
                self.assertEqual(cnt, 34, f"Real database was modified! Expected 34, got {cnt}")


if __name__ == "__main__":
    unittest.main()
