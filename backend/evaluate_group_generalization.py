"""
FraudLens — Isolated Unseen-Cardholder Evaluation Benchmark
===========================================================
Evaluates generalization to completely unseen cardholders using StratifiedGroupKFold
partitioning on credit card numbers (cc_num).

This is an isolated research evaluator. It DOES NOT overwrite or retrain the
production model saved at backend/model/fraud_detection_pipeline.pkl.

Methodology:
  1. Partitions backend/dataset/fraudTrain.csv using StratifiedGroupKFold(n_splits=5,
     shuffle=True, random_state=42) grouped on `cc_num`.
  2. Selects Fold 1 (786 training cardholders, 197 held-out testing cardholders; 0 overlap).
  3. Trains an isolated research pipeline with the exact architecture of train_model.py
     exclusively on the training cardholders.
  4. Evaluates the research pipeline against the held-out unseen cardholders under two
     distinct configurations:
       a. Full Feature Set Performance (all 11 features available)
       b. UI-Imputed Inference Performance (masked uncollected features: gender, lat, long,
          city_pop, merch_lat, merch_long replaced with np.nan, matching live app.py behavior)
  5. Updates backend/evaluation_report.json with an 'unseen_cardholder_benchmark' section
     while preserving existing random-split benchmark results.
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
from sklearn.model_selection import StratifiedGroupKFold
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


def run_unseen_cardholder_evaluation() -> dict:
    print("=" * 72)
    print("FraudLens — Isolated Unseen-Cardholder Generalization Benchmark")
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
    # 3. LOAD DATASET WITH GROUPING COLUMN
    # --------------------------------------------------
    cols = [
        "cc_num",
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

    # Feature engineering matching exact pipeline schema
    tx_time = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["transaction_hour"] = tx_time.dt.hour
    df["transaction_day"] = tx_time.dt.day
    df["transaction_month"] = tx_time.dt.month
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")

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

    X = df[feature_cols]
    y = df[target_col].astype(int)
    groups = df["cc_num"]

    total_unique_cardholders = int(groups.nunique())
    print(f"   Total rows: {total_dataset_rows:,} across {total_unique_cardholders:,} unique cardholders.")

    # --------------------------------------------------
    # 4. STRATIFIED GROUP K-FOLD PARTITIONING
    # --------------------------------------------------
    print("\n2. Partitioning dataset with StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train_group = X.iloc[train_idx]
    y_train_group = y.iloc[train_idx]
    train_groups = groups.iloc[train_idx]

    X_test_group = X.iloc[test_idx]
    y_test_group = y.iloc[test_idx]
    test_groups = groups.iloc[test_idx]

    train_cardholders = set(train_groups)
    test_cardholders = set(test_groups)
    train_unique_cardholders = len(train_cardholders)
    test_unique_cardholders = len(test_cardholders)
    cardholder_overlap = train_cardholders.intersection(test_cardholders)
    overlap_count = len(cardholder_overlap)

    if overlap_count != 0:
        raise ValueError(f"Partitioning error: Found {overlap_count} overlapping cardholders between train and test!")

    train_samples = len(train_idx)
    train_fraud_samples = int(y_train_group.sum())
    train_legit_samples = train_samples - train_fraud_samples
    train_fraud_pct = round(train_fraud_samples / train_samples * 100.0, 4)

    test_samples = len(test_idx)
    test_fraud_samples = int(y_test_group.sum())
    test_legit_samples = test_samples - test_fraud_samples
    test_fraud_pct = round(test_fraud_samples / test_samples * 100.0, 4)

    print(f"   Train Partition (Fold 1):")
    print(f"     - Rows:        {train_samples:,}")
    print(f"     - Fraud:       {train_fraud_samples:,} ({train_fraud_pct:.4f}%)")
    print(f"     - Legitimate:  {train_legit_samples:,}")
    print(f"     - Cardholders: {train_unique_cardholders:,}")
    print(f"   Test Partition (Fold 1):")
    print(f"     - Rows:        {test_samples:,}")
    print(f"     - Fraud:       {test_fraud_samples:,} ({test_fraud_pct:.4f}%)")
    print(f"     - Legitimate:  {test_legit_samples:,}")
    print(f"     - Cardholders: {test_unique_cardholders:,}")
    print(f"   Cardholder Overlap: {overlap_count} (Strictly Disjoint)")

    # Free raw dataframe memory
    del df, groups, train_groups, test_groups, train_cardholders, test_cardholders

    # --------------------------------------------------
    # 5. TRAIN ISOLATED RESEARCH PIPELINE
    # --------------------------------------------------
    print("\n3. Training isolated research pipeline on Fold 1 training cardholders...")
    print("   (Architecture: ColumnTransformer [SimpleImputer + OneHotEncoder] + RandomForestClassifier)")
    t_train_start = time.time()
    research_pipeline = build_isolated_pipeline(categorical_features, numerical_features)
    research_pipeline.fit(X_train_group, y_train_group)
    train_duration = round(time.time() - t_train_start, 2)
    print(f"   Isolated pipeline fitted in {train_duration}s")

    # Confirm production model was NOT modified
    if prod_mtime is not None:
        new_prod_mtime = os.path.getmtime(PRODUCTION_MODEL_PATH)
        assert new_prod_mtime == prod_mtime, "SAFETY VIOLATION: Production model file timestamp changed!"
        print("   Verified: Production model file was NOT touched.")

    # --------------------------------------------------
    # 6. BENCHMARK 1: FULL FEATURE SET ON UNSEEN CARDHOLDERS
    # --------------------------------------------------
    print("\n4. Running Benchmark A: Full Feature Set on Unseen Cardholders...")
    full_metrics = compute_metrics(research_pipeline, X_test_group, y_test_group, fraud_idx=1)
    print(f"   Full-feature inference completed in {full_metrics['inference_duration_sec']}s")
    print(
        f"   Precision: {full_metrics['precision']:.4f} | "
        f"Recall: {full_metrics['recall']:.4f} | "
        f"F1: {full_metrics['f1_score']:.4f} | "
        f"PR-AUC: {full_metrics['average_precision_pr_auc']:.4f} | "
        f"ROC-AUC: {full_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------
    # 7. BENCHMARK 2: UI-IMPUTED INFERENCE ON UNSEEN CARDHOLDERS
    # --------------------------------------------------
    print("\n5. Running Benchmark B: UI-Imputed Inference on Unseen Cardholders (masked uncollected features)...")
    masked_features = ["gender", "lat", "long", "city_pop", "merch_lat", "merch_long"]
    X_test_group_imputed = X_test_group.copy()
    for feat in masked_features:
        X_test_group_imputed[feat] = np.nan

    imputed_metrics = compute_metrics(research_pipeline, X_test_group_imputed, y_test_group, fraud_idx=1)
    print(f"   UI-imputed inference completed in {imputed_metrics['inference_duration_sec']}s")
    print(
        f"   Precision: {imputed_metrics['precision']:.4f} | "
        f"Recall: {imputed_metrics['recall']:.4f} | "
        f"F1: {imputed_metrics['f1_score']:.4f} | "
        f"PR-AUC: {imputed_metrics['average_precision_pr_auc']:.4f} | "
        f"ROC-AUC: {imputed_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------
    # 8. COMPARATIVE SUMMARY
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
    # 9. ASSEMBLE UNSEEN CARDHOLDER BENCHMARK OBJECT
    # --------------------------------------------------
    unseen_benchmark_data = {
        "evaluation_timestamp": datetime.now(IST_TZ).isoformat(),
        "split_strategy": {
            "method": "StratifiedGroupKFold",
            "group_column": "cc_num",
            "n_splits": 5,
            "selected_fold": 1,
            "random_state": 42,
            "total_unique_cardholders": total_unique_cardholders,
            "train_unique_cardholders": train_unique_cardholders,
            "test_unique_cardholders": test_unique_cardholders,
            "cardholder_overlap_count": overlap_count,
            "train_samples": train_samples,
            "train_fraud_samples": train_fraud_samples,
            "train_legitimate_samples": train_legit_samples,
            "train_fraud_prevalence_pct": train_fraud_pct,
            "test_samples": test_samples,
            "test_fraud_samples": test_fraud_samples,
            "test_legitimate_samples": test_legit_samples,
            "test_fraud_prevalence_pct": test_fraud_pct,
        },
        "model_metadata": {
            "pipeline_type": "Pipeline",
            "classifier": "RandomForestClassifier",
            "n_estimators": 100,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "training_split": "StratifiedGroupKFold Fold 1 (786 training cardholders, 197 unseen cardholders held out)",
            "is_production_model": False,
            "production_model_preserved": True,
            "training_duration_sec": train_duration,
        },
        "full_feature_evaluation": full_metrics,
        "ui_imputed_evaluation": {
            "description": "Simulation of UI inference conditions on unseen cardholders (uncollected features masked with np.nan and imputed by pipeline)",
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
            "unseen_cardholder_isolation": (
                "Card numbers (cc_num) were used as grouping identifiers in StratifiedGroupKFold. "
                "All 197 cardholders in the test partition were completely withheld from training (0 overlapping accounts). "
                "This measures the model's true capability to generalize to new, previously unseen cardholders."
            ),
            "masked_inference_features": (
                "Under UI inference, six features ('gender', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long') "
                "are not collected from the user. In backend/app.py, they are passed as numpy.nan to the pipeline's "
                "fitted SimpleImputer, which fills them with population training medians and modes. "
                "This benchmark quantifies the empirical impact of this imputation on held-out unseen cardholders."
            ),
        },
    }

    # --------------------------------------------------
    # 10. UPDATE EVALUATION REPORT FILE
    # --------------------------------------------------
    print(f"\n6. Updating {REPORT_PATH}...")
    existing_report = {}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load existing report: {exc}")

    existing_report["unseen_cardholder_benchmark"] = unseen_benchmark_data

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_report, f, indent=2)
    print(f"   Successfully updated {REPORT_PATH} with unseen_cardholder_benchmark.")

    # --------------------------------------------------
    # 11. DISPLAY FORMATTED COMPARISON TABLES
    # --------------------------------------------------
    print("\n" + "=" * 72)
    print("UNSEEN-CARDHOLDER BENCHMARK: FULL FEATURES vs. UI-IMPUTED INFERENCE")
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

    return unseen_benchmark_data


if __name__ == "__main__":
    run_unseen_cardholder_evaluation()
