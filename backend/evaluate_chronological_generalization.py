"""
FraudLens — Isolated Chronological Out-of-Time (OOT) Evaluation Benchmark
=========================================================================
Evaluates temporal generalization and concept drift on future transactions occurring
later in time than training data.

This is an isolated research evaluator. It DOES NOT overwrite or retrain the
production model saved at backend/model/fraud_detection_pipeline.pkl.

Methodology:
  1. Parses `trans_date_trans_time` as datetime and sorts backend/dataset/fraudTrain.csv
     strictly in chronological order.
  2. Uses the verified cutoff: `2020-03-06 07:16:43` (row 1,037,340).
     - Training partition: timestamps strictly earlier than cutoff (< 2020-03-06 07:16:43)
       -> 1,037,340 rows (5,968 fraud, 1,031,372 legitimate; 0.5753% prevalence).
     - Evaluation partition: timestamps at or after cutoff (>= 2020-03-06 07:16:43)
       -> 259,335 rows (1,538 fraud, 257,797 legitimate; 0.5931% prevalence).
  3. Trains an isolated research pipeline with the exact architecture of train_model.py
     exclusively on the training partition (all imputers and encoders fitted on train only).
  4. Evaluates the research pipeline against the chronological evaluation partition under:
       a. Full Feature Set Performance (all 11 features available)
       b. UI-Imputed Inference Performance (masked uncollected features: gender, lat, long,
          city_pop, merch_lat, merch_long replaced with np.nan, matching live app.py behavior)
  5. Updates backend/evaluation_report.json with a 'chronological_out_of_time_benchmark' section
     while preserving all existing random-split and group-held-out benchmark results.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

DATA_PATH = os.path.join(DATA_DIR, "fraudTrain.csv")
PRODUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_pipeline.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")

IST_TZ = timezone(timedelta(hours=5, minutes=30))
CUTOFF_STR = "2020-03-06 07:16:43"


def compute_metrics(pipeline, X_data: pd.DataFrame, y_data: pd.Series, fraud_idx: int = 1) -> dict:
    """Run pipeline prediction and compute evaluation metrics and threshold analysis."""
    t0 = time.time()
    y_pred = pipeline.predict(X_data)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_data)
        y_probs = probs[:, fraud_idx]
    else:
        y_probs = y_pred.astype(float)
    duration_sec = round(time.time() - t0, 2)

    prec_50 = float(precision_score(y_data, y_pred, zero_division=0))
    rec_50 = float(recall_score(y_data, y_pred, zero_division=0))
    f1_50 = float(f1_score(y_data, y_pred, zero_division=0))
    acc_50 = float((y_pred == y_data).mean())
    pr_auc = float(average_precision_score(y_data, y_probs))
    roc_auc = float(roc_auc_score(y_data, y_probs))

    cm = confusion_matrix(y_data, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    threshold_results = []
    threshold_configs = [
        (0.30, "Medium Risk Threshold (Flags suspicious transactions for verification)"),
        (0.50, "Standard Decision Boundary (Default binary classification cutoff)"),
        (0.70, "High Risk Threshold (Conservative cutoff minimizing false alarms)"),
    ]

    for thresh, desc in threshold_configs:
        pred_t = (y_probs >= thresh).astype(int)
        p_t = float(precision_score(y_data, pred_t, zero_division=0))
        r_t = float(recall_score(y_data, pred_t, zero_division=0))
        f_t = float(f1_score(y_data, pred_t, zero_division=0))
        cm_t = confusion_matrix(y_data, pred_t, labels=[0, 1])
        threshold_results.append({
            "threshold": thresh,
            "description": desc,
            "precision": round(p_t, 4),
            "recall": round(r_t, 4),
            "f1_score": round(f_t, 4),
            "true_positives": int(cm_t[1, 1]),
            "false_positives": int(cm_t[0, 1]),
            "false_negatives": int(cm_t[1, 0]),
            "true_negatives": int(cm_t[0, 0]),
        })

    return {
        "inference_duration_sec": duration_sec,
        "decision_threshold": 0.50,
        "precision": round(prec_50, 4),
        "recall": round(rec_50, 4),
        "f1_score": round(f1_50, 4),
        "accuracy": round(acc_50, 4),
        "average_precision_pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        },
        "threshold_analysis": threshold_results,
    }


def build_isolated_pipeline(categorical_features: list, numerical_features: list) -> Pipeline:
    """Build a fresh pipeline with architecture identical to production model."""
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numerical", numerical_transformer, numerical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def run_chronological_evaluation() -> dict:
    print("=" * 72)
    print("FraudLens — Isolated Chronological Out-of-Time Generalization Benchmark")
    print("=" * 72)

    # --------------------------------------------------
    # 2. VERIFY PRODUCTION MODEL ARTIFACT INTEGRITY
    # --------------------------------------------------
    if not os.path.exists(PRODUCTION_MODEL_PATH):
        print(f"Warning: Production model at {PRODUCTION_MODEL_PATH} not found.")
        prod_mtime = None
    else:
        prod_mtime = os.path.getmtime(PRODUCTION_MODEL_PATH)
        print(f"Verified production model intact at:\n  {PRODUCTION_MODEL_PATH} (mtime: {prod_mtime})")

    # --------------------------------------------------
    # 3. LOAD DATASET
    # --------------------------------------------------
    cols = [
        "trans_date_trans_time",
        "cc_num",
        "category",
        "gender",
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "is_fraud",
    ]

    print(f"\n1. Loading dataset from {DATA_PATH}...")
    t0 = time.time()
    try:
        df = pd.read_csv(DATA_PATH, usecols=cols)
        print(f"   Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    except Exception as exc:
        print(f"Error: Failed to read dataset: {exc}")
        sys.exit(1)

    target_col = "is_fraud"
    df = df.dropna(subset=[target_col])
    total_dataset_rows = len(df)

    # --------------------------------------------------
    # 4. PARSE DATETIME & SORT CHRONOLOGICALLY
    # --------------------------------------------------
    print("\n2. Parsing timestamps and sorting chronologically...")
    df["datetime"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    if df["datetime"].isna().sum() > 0:
        raise ValueError("Error: Found unparseable timestamps in dataset!")

    df = df.sort_values("datetime").reset_index(drop=True)

    min_timestamp = df["datetime"].min()
    max_timestamp = df["datetime"].max()
    print(f"   Timeline range: {min_timestamp} -> {max_timestamp}")

    # --------------------------------------------------
    # 5. CHRONOLOGICAL PARTITIONING
    # --------------------------------------------------
    cutoff = pd.to_datetime(CUTOFF_STR)
    print(f"\n3. Applying chronological cutoff at {cutoff}...")

    train_mask = df["datetime"] < cutoff
    test_mask = df["datetime"] >= cutoff

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    train_samples = len(train_df)
    test_samples = len(test_df)
    train_fraud_samples = int(train_df[target_col].sum())
    test_fraud_samples = int(test_df[target_col].sum())
    train_legit_samples = train_samples - train_fraud_samples
    test_legit_samples = test_samples - test_fraud_samples
    train_fraud_pct = round(train_fraud_samples / train_samples * 100.0, 4)
    test_fraud_pct = round(test_fraud_samples / test_samples * 100.0, 4)

    # Strict integrity assertions against audit findings
    if train_samples != 1037340:
        raise ValueError(f"Partition discrepancy: expected 1,037,340 train rows, got {train_samples}")
    if test_samples != 259335:
        raise ValueError(f"Partition discrepancy: expected 259,335 test rows, got {test_samples}")
    if train_fraud_samples != 5968:
        raise ValueError(f"Fraud count discrepancy: expected 5,968 train frauds, got {train_fraud_samples}")
    if test_fraud_samples != 1538:
        raise ValueError(f"Fraud count discrepancy: expected 1,538 test frauds, got {test_fraud_samples}")

    train_max_dt = train_df["datetime"].max()
    test_min_dt = test_df["datetime"].min()
    if train_max_dt >= cutoff:
        raise ValueError(f"Temporal leakage violation: train max {train_max_dt} >= cutoff {cutoff}")
    if test_min_dt < cutoff:
        raise ValueError(f"Temporal leakage violation: test min {test_min_dt} < cutoff {cutoff}")

    train_cards = set(train_df["cc_num"])
    test_cards = set(test_df["cc_num"])
    card_overlap = train_cards.intersection(test_cards)
    overlap_count = len(card_overlap)
    overlap_pct = round(overlap_count / len(test_cards) * 100.0, 2)

    print(f"   Training Partition (< {cutoff}):")
    print(f"     - Rows:            {train_samples:,} ({train_samples/total_dataset_rows*100:.2f}%)")
    print(f"     - Time Span:       {min_timestamp} -> {train_max_dt}")
    print(f"     - Fraud:           {train_fraud_samples:,} ({train_fraud_pct:.4f}%)")
    print(f"     - Legitimate:      {train_legit_samples:,}")
    print(f"     - Unique Cards:    {len(train_cards):,}")
    print(f"   Evaluation Partition (>= {cutoff}):")
    print(f"     - Rows:            {test_samples:,} ({test_samples/total_dataset_rows*100:.2f}%)")
    print(f"     - Time Span:       {test_min_dt} -> {max_timestamp}")
    print(f"     - Fraud:           {test_fraud_samples:,} ({test_fraud_pct:.4f}%)")
    print(f"     - Legitimate:      {test_legit_samples:,}")
    print(f"     - Unique Cards:    {len(test_cards):,}")
    print(f"   Active Cardholder Overlap: {overlap_count:,} ({overlap_pct}% of test accounts active in train)")

    # --------------------------------------------------
    # 6. FEATURE ENGINEERING (INDEPENDENT SINGLE-ROW)
    # --------------------------------------------------
    for d in [train_df, test_df]:
        d["transaction_hour"] = d["datetime"].dt.hour
        d["transaction_day"] = d["datetime"].dt.day
        d["transaction_month"] = d["datetime"].dt.month
        d["amt"] = pd.to_numeric(d["amt"], errors="coerce")

    categorical_features = ["category", "gender"]
    numerical_features = [
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
    feature_cols = categorical_features + numerical_features

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    del df, train_df, test_df

    # --------------------------------------------------
    # 7. TRAIN ISOLATED RESEARCH PIPELINE
    # --------------------------------------------------
    print("\n4. Training isolated research pipeline on chronological training partition...")
    print("   (Imputers, encoders, and Random Forest fitted EXCLUSIVELY on training data)")
    t_train_start = time.time()
    research_pipeline = build_isolated_pipeline(categorical_features, numerical_features)
    research_pipeline.fit(X_train, y_train)
    train_duration = round(time.time() - t_train_start, 2)
    print(f"   Isolated pipeline fitted in {train_duration}s")

    # Confirm production model was NOT modified
    if prod_mtime is not None:
        new_prod_mtime = os.path.getmtime(PRODUCTION_MODEL_PATH)
        assert new_prod_mtime == prod_mtime, "SAFETY VIOLATION: Production model file timestamp changed!"
        print("   Verified: Production model file was NOT touched.")

    # --------------------------------------------------
    # 8. BENCHMARK 1: FULL FEATURE SET ON FUTURE TRANSACTIONS
    # --------------------------------------------------
    print("\n5. Running Benchmark A: Full Feature Set on Future Transactions...")
    full_metrics = compute_metrics(research_pipeline, X_test, y_test, fraud_idx=1)
    print(f"   Full-feature inference completed in {full_metrics['inference_duration_sec']}s")
    print(
        f"   Precision: {full_metrics['precision']:.4f} | "
        f"Recall: {full_metrics['recall']:.4f} | "
        f"F1: {full_metrics['f1_score']:.4f} | "
        f"PR-AUC: {full_metrics['average_precision_pr_auc']:.4f} | "
        f"ROC-AUC: {full_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------
    # 9. BENCHMARK 2: UI-IMPUTED INFERENCE ON FUTURE TRANSACTIONS
    # --------------------------------------------------
    print("\n6. Running Benchmark B: UI-Imputed Inference on Future Transactions (masked uncollected features)...")
    masked_features = ["gender", "lat", "long", "city_pop", "merch_lat", "merch_long"]
    X_test_imputed = X_test.copy()
    for feat in masked_features:
        X_test_imputed[feat] = np.nan

    imputed_metrics = compute_metrics(research_pipeline, X_test_imputed, y_test, fraud_idx=1)
    print(f"   UI-imputed inference completed in {imputed_metrics['inference_duration_sec']}s")
    print(
        f"   Precision: {imputed_metrics['precision']:.4f} | "
        f"Recall: {imputed_metrics['recall']:.4f} | "
        f"F1: {imputed_metrics['f1_score']:.4f} | "
        f"PR-AUC: {imputed_metrics['average_precision_pr_auc']:.4f} | "
        f"ROC-AUC: {imputed_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------
    # 10. COMPARATIVE SUMMARY
    # --------------------------------------------------
    cm_full = full_metrics["confusion_matrix"]
    cm_imp = imputed_metrics["confusion_matrix"]

    comp_summary = {
        "evaluated_samples": test_samples,
        "fraud_samples": test_fraud_samples,
        "legitimate_samples": test_legit_samples,
        "fraud_prevalence_pct": test_fraud_pct,
        "metrics_comparison": {
            "precision": {
                "full_features": full_metrics["precision"],
                "ui_imputed": imputed_metrics["precision"],
                "delta": round(imputed_metrics["precision"] - full_metrics["precision"], 4),
            },
            "recall": {
                "full_features": full_metrics["recall"],
                "ui_imputed": imputed_metrics["recall"],
                "delta": round(imputed_metrics["recall"] - full_metrics["recall"], 4),
            },
            "f1_score": {
                "full_features": full_metrics["f1_score"],
                "ui_imputed": imputed_metrics["f1_score"],
                "delta": round(imputed_metrics["f1_score"] - full_metrics["f1_score"], 4),
            },
            "accuracy": {
                "full_features": full_metrics["accuracy"],
                "ui_imputed": imputed_metrics["accuracy"],
                "delta": round(imputed_metrics["accuracy"] - full_metrics["accuracy"], 4),
            },
            "average_precision_pr_auc": {
                "full_features": full_metrics["average_precision_pr_auc"],
                "ui_imputed": imputed_metrics["average_precision_pr_auc"],
                "delta": round(imputed_metrics["average_precision_pr_auc"] - full_metrics["average_precision_pr_auc"], 4),
            },
            "roc_auc": {
                "full_features": full_metrics["roc_auc"],
                "ui_imputed": imputed_metrics["roc_auc"],
                "delta": round(imputed_metrics["roc_auc"] - full_metrics["roc_auc"], 4),
            },
        },
        "confusion_matrix_comparison": {
            "true_positives": {
                "full_features": cm_full["true_positives"],
                "ui_imputed": cm_imp["true_positives"],
                "delta": cm_imp["true_positives"] - cm_full["true_positives"],
            },
            "false_positives": {
                "full_features": cm_full["false_positives"],
                "ui_imputed": cm_imp["false_positives"],
                "delta": cm_imp["false_positives"] - cm_full["false_positives"],
            },
            "false_negatives": {
                "full_features": cm_full["false_negatives"],
                "ui_imputed": cm_imp["false_negatives"],
                "delta": cm_imp["false_negatives"] - cm_full["false_negatives"],
            },
            "true_negatives": {
                "full_features": cm_full["true_negatives"],
                "ui_imputed": cm_imp["true_negatives"],
                "delta": cm_imp["true_negatives"] - cm_full["true_negatives"],
            },
        },
    }

    # --------------------------------------------------
    # 11. ASSEMBLE CHRONOLOGICAL BENCHMARK OBJECT
    # --------------------------------------------------
    chronological_benchmark_data = {
        "evaluation_timestamp": datetime.now(IST_TZ).isoformat(),
        "split_strategy": {
            "method": "Chronological Holdout (Out-of-Time)",
            "cutoff_timestamp": CUTOFF_STR,
            "training_start_timestamp": str(min_timestamp),
            "training_end_timestamp": str(train_max_dt),
            "evaluation_start_timestamp": str(test_min_dt),
            "evaluation_end_timestamp": str(max_timestamp),
            "total_dataset_rows": total_dataset_rows,
            "train_samples": train_samples,
            "train_fraud_samples": train_fraud_samples,
            "train_legitimate_samples": train_legit_samples,
            "train_fraud_prevalence_pct": train_fraud_pct,
            "test_samples": test_samples,
            "test_fraud_samples": test_fraud_samples,
            "test_legitimate_samples": test_legit_samples,
            "test_fraud_prevalence_pct": test_fraud_pct,
            "train_unique_cardholders": len(train_cards),
            "test_unique_cardholders": len(test_cards),
            "cardholder_overlap_count": overlap_count,
            "cardholder_overlap_pct_of_test": overlap_pct,
            "cardholder_overlap_disclosure": (
                "Descriptive statistic illustrating ongoing customer spending over time. "
                "This benchmark tests temporal stability and drift, not zero-shot cardholder generalization."
            ),
        },
        "model_metadata": {
            "pipeline_type": "Pipeline",
            "classifier": "RandomForestClassifier",
            "n_estimators": 100,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "training_split": f"Transactions strictly prior to {CUTOFF_STR}",
            "is_production_model": False,
            "production_model_preserved": True,
            "preprocessing_fitted_on_train_only": True,
            "training_duration_sec": train_duration,
        },
        "full_feature_evaluation": full_metrics,
        "ui_imputed_evaluation": {
            "description": "Simulation of UI inference conditions on future transactions (uncollected features masked with np.nan and imputed by pipeline)",
            "masked_features": masked_features,
            **imputed_metrics,
        },
        "comparative_summary": comp_summary,
        "methodology_disclosures": {
            "offline_simulation_disclosure": (
                "IMPORTANT: These are offline research benchmark evaluations on synthetic transaction dataset partitions, "
                "not verified performance metrics on real-world bank, credit card, or UPI transactions. "
                "The UI-imputed evaluation simulates the exact feature availability of the live web application "
                "by masking uncollected inputs with numpy.nan, but does not prove live production efficacy."
            ),
            "chronological_out_of_time_separation": (
                f"All training transactions strictly precede {CUTOFF_STR}; all evaluation transactions occur at or after this cutoff. "
                "Feature engineering, imputers, and encoders were fitted exclusively on the training partition without looking ahead."
            ),
            "cardholder_overlap_distinction": (
                "Unlike the group-held-out benchmark (which isolates accounts with 0 overlap), this chronological benchmark naturally "
                f"includes repeated cardholders across time ({overlap_count} overlapping cardholders, {overlap_pct}% of test accounts), "
                "evaluating temporal robustness, behavioral drift, and seasonality rather than zero-shot account generalization."
            ),
            "masked_inference_features": (
                "Under UI inference, six features ('gender', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long') "
                "are not collected from the user. In backend/app.py, they are passed as numpy.nan to the pipeline's "
                "fitted SimpleImputer, which fills them with population training medians and modes. "
                "This benchmark quantifies the empirical impact of this imputation on chronologically future transactions."
            ),
        },
    }

    # --------------------------------------------------
    # 12. UPDATE EVALUATION REPORT FILE
    # --------------------------------------------------
    print(f"\n7. Updating {REPORT_PATH}...")
    existing_report = {}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load existing report: {exc}")

    existing_report["chronological_out_of_time_benchmark"] = chronological_benchmark_data

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_report, f, indent=2)
    print(f"   Successfully updated {REPORT_PATH} with chronological_out_of_time_benchmark.")

    # --------------------------------------------------
    # 13. DISPLAY FORMATTED COMPARISON TABLES
    # --------------------------------------------------
    print("\n" + "=" * 72)
    print("CHRONOLOGICAL OUT-OF-TIME BENCHMARK: FULL FEATURES vs. UI-IMPUTED INFERENCE")
    print("=" * 72)
    print(f"{'Metric':<26} | {'Full Features':<16} | {'UI-Imputed (np.nan)':<19} | {'Delta':<8}")
    print("-" * 72)
    print(f"{'Precision':<26} | {full_metrics['precision'] * 100:>14.2f}% | {imputed_metrics['precision'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['precision']['delta'] * 100:>+7.2f}%")
    print(f"{'Recall':<26} | {full_metrics['recall'] * 100:>14.2f}% | {imputed_metrics['recall'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['recall']['delta'] * 100:>+7.2f}%")
    print(f"{'F1 Score':<26} | {full_metrics['f1_score']:>15.4f} | {imputed_metrics['f1_score']:>18.4f} | {comp_summary['metrics_comparison']['f1_score']['delta']:>+8.4f}")
    print(f"{'Accuracy':<26} | {full_metrics['accuracy'] * 100:>14.2f}% | {imputed_metrics['accuracy'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['accuracy']['delta'] * 100:>+7.2f}%")
    print(f"{'PR-AUC (Avg Precision)':<26} | {full_metrics['average_precision_pr_auc']:>15.4f} | {imputed_metrics['average_precision_pr_auc']:>18.4f} | {comp_summary['metrics_comparison']['average_precision_pr_auc']['delta']:>+8.4f}")
    print(f"{'ROC-AUC':<26} | {full_metrics['roc_auc']:>15.4f} | {imputed_metrics['roc_auc']:>18.4f} | {comp_summary['metrics_comparison']['roc_auc']['delta']:>+8.4f}")
    print("-" * 72)
    print(f"{'True Positives (TP)':<26} | {cm_full['true_positives']:>15,d} | {cm_imp['true_positives']:>18,d} | {comp_summary['confusion_matrix_comparison']['true_positives']['delta']:>+8,d}")
    print(f"{'False Positives (FP)':<26} | {cm_full['false_positives']:>15,d} | {cm_imp['false_positives']:>18,d} | {comp_summary['confusion_matrix_comparison']['false_positives']['delta']:>+8,d}")
    print(f"{'False Negatives (FN)':<26} | {cm_full['false_negatives']:>15,d} | {cm_imp['false_negatives']:>18,d} | {comp_summary['confusion_matrix_comparison']['false_negatives']['delta']:>+8,d}")
    print(f"{'True Negatives (TN)':<26} | {cm_full['true_negatives']:>15,d} | {cm_imp['true_negatives']:>18,d} | {comp_summary['confusion_matrix_comparison']['true_negatives']['delta']:>+8,d}")
    print("=" * 72)

    return chronological_benchmark_data


if __name__ == "__main__":
    run_chronological_evaluation()
