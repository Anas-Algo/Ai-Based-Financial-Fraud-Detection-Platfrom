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

from app import DB_PATH, app, model
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

    def test_13_historical_dataset_timestamp_distinguished_from_analysis_time(self):
        """Verify historical payment timestamp is distinguished from analysis time."""
        payload = {
            "transactionId": "TXN-HISTORICAL-01",
            "amount": 180.0,
            "channel": "Card",
            "merchantCategory": "shopping_net",
            "transactionTime": "2019-01-02 01:06:37",
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["transactionId"], "TXN-HISTORICAL-01")
        self.assertEqual(data["timestamp"], "2019-01-02 01:06:37")
        self.assertEqual(data["transactionTime"], "2019-01-02 01:06:37")
        self.assertTrue(data["hasTransactionTime"])
        self.assertIn("createdAt", data)
        # Analysis time is contemporary (not 2019)
        self.assertNotIn("2019", data["createdAt"])

        # Check /api/transactions persistence
        hist_res = self.client.get("/api/transactions")
        self.assertEqual(hist_res.status_code, 200)
        txns = hist_res.get_json()["transactions"]
        match = next((t for t in txns if t["transactionId"] == "TXN-HISTORICAL-01"), None)
        self.assertIsNotNone(match)
        self.assertEqual(match["timestamp"], "2019-01-02 01:06:37")
        self.assertTrue(match["hasTransactionTime"])

    def test_14_manually_submitted_transaction_missing_timestamp(self):
        """Verify transaction submitted without payment time is marked as analysis time."""
        payload = {
            "transactionId": "TXN-MANUAL-NO-TIME-01",
            "amount": 95.0,
            "channel": "UPI",
            "merchantCategory": "Groceries",
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
        """Verify that missing uncollected features (gender, lat, long, etc.) produce valid finite predictions."""
        payload = {
            "transactionId": "TXN-IMPUTE-TEST-01",
            "amount": 125.75,
            "channel": "UPI",
            "merchantCategory": "Restaurants",
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

    def test_16_imputation_handles_explicit_demographic_and_geographic_inputs(self):
        """Verify that explicit inputs for uncollected features are accepted and processed when provided."""
        payload = {
            "transactionId": "TXN-EXPLICIT-COORDS-01",
            "amount": 75.0,
            "channel": "Card",
            "merchantCategory": "Entertainment",
            "gender": "F",
            "lat": 40.7128,
            "long": -74.0060,
            "city_pop": 8000000,
            "merch_lat": 40.7130,
            "merch_long": -74.0065,
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["transactionId"], "TXN-EXPLICIT-COORDS-01")
        self.assertTrue(math.isfinite(data["fraud_probability"]))
        self.assertTrue(math.isfinite(data["confidence"]))

    def test_17_pipeline_fitted_imputer_statistics(self):
        """Verify the pipeline's actual SimpleImputer statistics match dataset learned medians and modes."""
        self.assertIsNotNone(model, "Model pipeline should be loaded")
        preprocessor = model.named_steps["preprocessor"]

        # 1. Numerical imputer verification
        num_imputer = preprocessor.named_transformers_["numerical"].named_steps["imputer"]
        stats = num_imputer.statistics_
        # Expected features: ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'transaction_hour', 'transaction_day', 'transaction_month']
        self.assertAlmostEqual(float(stats[1]), 39.3543, places=3)   # lat
        self.assertAlmostEqual(float(stats[2]), -87.4769, places=3)  # long
        self.assertEqual(float(stats[3]), 2456.0)                   # city_pop
        self.assertAlmostEqual(float(stats[4]), 39.3705, places=3)   # merch_lat
        self.assertAlmostEqual(float(stats[5]), -87.4411, places=3)  # merch_long

        # 2. Categorical imputer verification
        cat_imputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]
        cat_stats = cat_imputer.statistics_
        # Expected features: ['category', 'gender']
        self.assertEqual(str(cat_stats[1]), "F")                    # gender mode

        # 3. Transform verification with np.nan
        sample_df = pd.DataFrame([{
            "category": "grocery_pos",
            "gender": np.nan,
            "amt": 50.0,
            "lat": np.nan,
            "long": np.nan,
            "city_pop": np.nan,
            "merch_lat": np.nan,
            "merch_long": np.nan,
            "transaction_hour": 14,
            "transaction_day": 15,
            "transaction_month": 6,
        }])
        transformed = preprocessor.transform(sample_df)
        self.assertEqual(transformed.shape, (1, 25))
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


if __name__ == "__main__":
    unittest.main()
