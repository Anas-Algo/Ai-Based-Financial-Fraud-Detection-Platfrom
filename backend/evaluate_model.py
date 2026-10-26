"""
FraudLens — Offline Machine Learning Model Evaluation
=====================================================
Evaluates the pre-trained Scikit-learn Random Forest fraud detection pipeline
against a held-out evaluation dataset under two distinct configurations:
  1. Full Feature Set Performance (all 11 features available)
  2. UI-Imputed Inference Performance (uncollected features masked with np.nan,
     matching live inference behavior in backend/app.py)

Outputs:
  - backend/evaluation_report.json: Serialized comparative metrics, confusion matrices,
    threshold breakdown, and methodology disclosures.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_pipeline.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "fraudTest.csv")

IST_TZ = timezone(timedelta(hours=5, minutes=30))


def compute_metrics(pipeline, X_data: pd.DataFrame, y_data: pd.Series, fraud_idx: int) -> dict:
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


def run_evaluation() -> dict:
    print("=" * 68)
    print("FraudLens — Offline Model Evaluation & UI Inference Benchmark")
    print("=" * 68)

    # --------------------------------------------------
    # 2. VALIDATE REQUIRED ARTIFACTS
    # --------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Saved model pipeline not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"\n1. Loading saved model pipeline from:\n   {MODEL_PATH}")
    t0 = time.time()
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"   Model pipeline loaded in {time.time() - t0:.2f}s")
    except Exception as exc:
        print(f"Error: Failed to load model pipeline: {exc}")
        sys.exit(1)

    # Validate pipeline steps
    if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
        print("Error: Pipeline does not contain the expected 'classifier' step.")
        sys.exit(1)

    classifier = pipeline.named_steps["classifier"]
    classes = list(getattr(classifier, "classes_", [0, 1]))
    fraud_idx = classes.index(1) if 1 in classes else 1
    print(f"   Classifier: {type(classifier).__name__} (n_estimators={getattr(classifier, 'n_estimators', None)})")
    print(f"   Classes: {classes} (Positive fraud index: {fraud_idx})")

    # --------------------------------------------------
    # 3. SELECT EVALUATION DATASET
    # --------------------------------------------------
    using_separate_test_file = False

    if os.path.exists(TEST_DATA_PATH) and os.path.getsize(TEST_DATA_PATH) > 0:
        data_file_path = TEST_DATA_PATH
        eval_dataset_source = "fraudTest.csv (independent benchmark test file)"
        using_separate_test_file = True
    elif os.path.exists(TRAIN_DATA_PATH) and os.path.getsize(TRAIN_DATA_PATH) > 0:
        data_file_path = TRAIN_DATA_PATH
        eval_dataset_source = (
            "fraudTrain.csv (held-out 20% test partition withheld from pipeline.fit() "
            "using train_test_split test_size=0.20, random_state=42, stratify=y)"
        )
        using_separate_test_file = False
    else:
        print(f"Error: No valid evaluation dataset found in {DATA_DIR}.")
        sys.exit(1)

    print(f"\n2. Evaluation Dataset Source:\n   {eval_dataset_source}")
    print(f"   File: {data_file_path}")

    # --------------------------------------------------
    # 4. LOAD AND PREPARE FEATURES
    # --------------------------------------------------
    cols = [
        "trans_date_trans_time",
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

    print("\n3. Loading evaluation features...")
    t1 = time.time()
    try:
        df = pd.read_csv(data_file_path, usecols=cols)
        print(f"   Loaded {len(df):,} rows in {time.time() - t1:.2f}s")
    except Exception as exc:
        print(f"Error: Failed to read dataset: {exc}")
        sys.exit(1)

    target_col = "is_fraud"
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' missing from dataset.")
        sys.exit(1)

    df = df.dropna(subset=[target_col])

    # Feature engineering matching exact pipeline training schema
    tx_time = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["transaction_hour"] = tx_time.dt.hour
    df["transaction_day"] = tx_time.dt.day
    df["transaction_month"] = tx_time.dt.month
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")

    feature_cols = [
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

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # --------------------------------------------------
    # 5. FORM HELD-OUT TEST PARTITION
    # --------------------------------------------------
    total_dataset_rows = len(df)
    if using_separate_test_file:
        X_eval = X
        y_eval = y
    else:
        print("\n4. Reconstructing exact 20% held-out test split (test_size=0.20, random_state=42, stratify=y)...")
        _, X_eval, _, y_eval = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

    # Free memory
    del df, X, y

    total_eval_samples = len(X_eval)
    fraud_eval_samples = int(y_eval.sum())
    legit_eval_samples = total_eval_samples - fraud_eval_samples
    fraud_prevalence_pct = round(fraud_eval_samples / total_eval_samples * 100.0, 4)

    print(f"   Evaluation samples: {total_eval_samples:,}")
    print(f"   Legitimate:        {legit_eval_samples:,} ({100.0 - fraud_prevalence_pct:.2f}%)")
    print(f"   Fraudulent:        {fraud_eval_samples:,} ({fraud_prevalence_pct:.4f}%)")

    # --------------------------------------------------
    # 6. BENCHMARK 1: FULL FEATURE SET PERFORMANCE
    # --------------------------------------------------
    print("\n5. Running Benchmark 1: Full Feature Set Performance...")
    full_metrics = compute_metrics(pipeline, X_eval, y_eval, fraud_idx)
    print(f"   Full-feature inference completed in {full_metrics['inference_duration_sec']}s")
    print(f"   Precision: {full_metrics['precision']:.4f} | Recall: {full_metrics['recall']:.4f} | F1: {full_metrics['f1_score']:.4f} | PR-AUC: {full_metrics['average_precision_pr_auc']:.4f}")

    # --------------------------------------------------
    # 7. BENCHMARK 2: UI-IMPUTED INFERENCE PERFORMANCE
    # --------------------------------------------------
    print("\n6. Running Benchmark 2: UI-Imputed Inference Performance (masked uncollected features)...")
    masked_features = ["gender", "lat", "long", "city_pop", "merch_lat", "merch_long"]
    X_eval_imputed = X_eval.copy()
    for feat in masked_features:
        X_eval_imputed[feat] = np.nan

    imputed_metrics = compute_metrics(pipeline, X_eval_imputed, y_eval, fraud_idx)
    print(f"   UI-imputed inference completed in {imputed_metrics['inference_duration_sec']}s")
    print(f"   Precision: {imputed_metrics['precision']:.4f} | Recall: {imputed_metrics['recall']:.4f} | F1: {imputed_metrics['f1_score']:.4f} | PR-AUC: {imputed_metrics['average_precision_pr_auc']:.4f}")

    # --------------------------------------------------
    # 8. CONSTRUCT COMPARATIVE SUMMARY
    # --------------------------------------------------
    cm_full = full_metrics["confusion_matrix"]
    cm_imp = imputed_metrics["confusion_matrix"]

    comp_summary = {
        "evaluated_samples": total_eval_samples,
        "fraud_samples": fraud_eval_samples,
        "legitimate_samples": legit_eval_samples,
        "fraud_prevalence_pct": fraud_prevalence_pct,
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
    # 9. CONSTRUCT STRUCTURED REPORT
    # --------------------------------------------------
    report_data = {
        "evaluation_timestamp": datetime.now(IST_TZ).isoformat(),
        "model_metadata": {
            "model_path": os.path.relpath(MODEL_PATH, BASE_DIR).replace("\\", "/"),
            "pipeline_type": type(pipeline).__name__,
            "classifier": type(classifier).__name__,
            "n_estimators": getattr(classifier, "n_estimators", 100),
            "class_weight": str(getattr(classifier, "class_weight", "balanced_subsample")),
            "random_state": getattr(classifier, "random_state", 42),
            "classes": [int(c) for c in classes],
            "total_preprocessed_features": getattr(classifier, "n_features_in_", 25),
        },
        "dataset_metadata": {
            "source_file": os.path.relpath(data_file_path, BASE_DIR).replace("\\", "/"),
            "evaluation_split_method": eval_dataset_source,
            "using_separate_test_file": using_separate_test_file,
            "total_dataset_rows": total_dataset_rows,
            "evaluation_samples": total_eval_samples,
            "fraud_samples": fraud_eval_samples,
            "legitimate_samples": legit_eval_samples,
            "fraud_prevalence_pct": fraud_prevalence_pct,
        },
        "feature_schema": {
            "raw_input_features": feature_cols,
            "categorical_features": ["category", "gender"],
            "numerical_features": [
                "amt",
                "lat",
                "long",
                "city_pop",
                "merch_lat",
                "merch_long",
                "transaction_hour",
                "transaction_day",
                "transaction_month",
            ],
            "inference_form_inputs": ["amount", "channel", "merchantCategory", "country"],
            "uncollected_masked_features": masked_features,
            "imputation_handling": "np.nan passed into pipeline; fitted SimpleImputer supplies learned training medians & modes",
        },
        "full_feature_evaluation": full_metrics,
        "ui_imputed_evaluation": {
            "description": "Simulation of UI inference conditions where uncollected features are masked with np.nan and imputed by pipeline",
            "masked_features": masked_features,
            **imputed_metrics,
        },
        "comparative_summary": comp_summary,
        # Preserve top-level evaluation_metrics for backward compatibility
        "evaluation_metrics": full_metrics,
        "threshold_analysis": full_metrics["threshold_analysis"],
        "methodology_disclosures": {
            "offline_simulation_disclosure": (
                "IMPORTANT: These are offline benchmark evaluations on synthetic transaction dataset partitions, "
                "not verified performance metrics on real-world bank, credit card, or UPI transactions. "
                "The UI-imputed evaluation simulates the exact feature availability of the live web application "
                "by masking uncollected inputs with numpy.nan, but does not prove live production efficacy."
            ),
            "class_imbalance": (
                "Severe positive class imbalance (0.579% fraud). Accuracy alone (99.8%) "
                "is deceptive because a trivial majority-class predictor achieves 99.42% accuracy. "
                "Precision, Recall, and PR-AUC are the primary reliability benchmarks."
            ),
            "masked_inference_features": (
                "Under UI inference, six features ('gender', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long') "
                "are not collected from the user. In backend/app.py, they are passed as numpy.nan to the pipeline's "
                "fitted SimpleImputer, which fills them with population training medians and modes. "
                "This benchmark quantifies the empirical impact of this imputation on held-out test data."
            ),
            "temporal_split_limitation": (
                "The held-out evaluation set was partitioned using stratified random sampling "
                "(test_size=0.20, random_state=42), matching train_model.py. Transactions from future "
                "dates exist in both partitions. An out-of-time (OOT) evaluation on subsequent dates is "
                "recommended for strict temporal validation."
            ),
            "account_grouping_limitation": (
                "Card numbers (cc_num) were withheld from training features but not grouped during splitting. "
                "Transactions from the same synthetic customer appear in both train and test partitions."
            ),
        },
    }

    # --------------------------------------------------
    # 10. SAVE REPORT
    # --------------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # --------------------------------------------------
    # 11. DISPLAY FORMATTED COMPARISON TABLE
    # --------------------------------------------------
    print("\n" + "=" * 68)
    print("COMPARATIVE BENCHMARK: FULL FEATURES vs. UI-IMPUTED INFERENCE")
    print("=" * 68)
    print(f"{'Metric':<26} | {'Full Features':<16} | {'UI-Imputed (np.nan)':<19} | {'Delta':<8}")
    print("-" * 68)
    print(f"{'Precision':<26} | {full_metrics['precision'] * 100:>14.2f}% | {imputed_metrics['precision'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['precision']['delta'] * 100:>+7.2f}%")
    print(f"{'Recall':<26} | {full_metrics['recall'] * 100:>14.2f}% | {imputed_metrics['recall'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['recall']['delta'] * 100:>+7.2f}%")
    print(f"{'F1 Score':<26} | {full_metrics['f1_score']:>15.4f} | {imputed_metrics['f1_score']:>18.4f} | {comp_summary['metrics_comparison']['f1_score']['delta']:>+8.4f}")
    print(f"{'Accuracy':<26} | {full_metrics['accuracy'] * 100:>14.2f}% | {imputed_metrics['accuracy'] * 100:>17.2f}% | {comp_summary['metrics_comparison']['accuracy']['delta'] * 100:>+7.2f}%")
    print(f"{'PR-AUC (Avg Precision)':<26} | {full_metrics['average_precision_pr_auc']:>15.4f} | {imputed_metrics['average_precision_pr_auc']:>18.4f} | {comp_summary['metrics_comparison']['average_precision_pr_auc']['delta']:>+8.4f}")
    print(f"{'ROC-AUC':<26} | {full_metrics['roc_auc']:>15.4f} | {imputed_metrics['roc_auc']:>18.4f} | {comp_summary['metrics_comparison']['roc_auc']['delta']:>+8.4f}")
    print("-" * 68)
    print(f"{'True Positives (TP)':<26} | {cm_full['true_positives']:>15,d} | {cm_imp['true_positives']:>18,d} | {comp_summary['confusion_matrix_comparison']['true_positives']['delta']:>+8,d}")
    print(f"{'False Positives (FP)':<26} | {cm_full['false_positives']:>15,d} | {cm_imp['false_positives']:>18,d} | {comp_summary['confusion_matrix_comparison']['false_positives']['delta']:>+8,d}")
    print(f"{'False Negatives (FN)':<26} | {cm_full['false_negatives']:>15,d} | {cm_imp['false_negatives']:>18,d} | {comp_summary['confusion_matrix_comparison']['false_negatives']['delta']:>+8,d}")
    print(f"{'True Negatives (TN)':<26} | {cm_full['true_negatives']:>15,d} | {cm_imp['true_negatives']:>18,d} | {comp_summary['confusion_matrix_comparison']['true_negatives']['delta']:>+8,d}")
    print("=" * 68)
    print(f"Report saved to:\n  {REPORT_PATH}")
    print("=" * 68)

    return report_data


if __name__ == "__main__":
    run_evaluation()
