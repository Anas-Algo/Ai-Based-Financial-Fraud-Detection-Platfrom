"""
FraudLens — Indian UPI Model Training Pipeline
==============================================
Trains a compact, depth-constrained RandomForestClassifier pipeline on the
synthetic Indian UPI transaction baseline.

ACADEMIC SIMULATION NOTICE:
---------------------------
This training pipeline is part of an academic prototype simulation.
It trains on a synthetic UPI baseline and does NOT evaluate real-world
bank clearing networks or live NPCI switch traffic.
"""

import os
import sys
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

DATA_PATH = os.path.join(DATA_DIR, "synthetic_baseline_inr.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "fraudlens_india_pipeline.pkl")


def load_or_generate_dataset() -> pd.DataFrame:
    """Load the synthetic baseline dataset, generating it deterministically if absent."""
    if not os.path.exists(DATA_PATH):
        print(f"Synthetic baseline not found at {DATA_PATH}. Generating deterministic baseline (seed=42)...")
        sys.path.insert(0, DATA_DIR)
        from generate_synthetic_baseline import generate_synthetic_data, DEFAULT_OUTPUT_PATH
        df = generate_synthetic_data(n_samples=50000, random_state=42)
        os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
        df.to_csv(DEFAULT_OUTPUT_PATH, index=False)
        print(f"Generated and saved 50,000 rows to {DEFAULT_OUTPUT_PATH}")
        return df

    print(f"Loading synthetic baseline from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset shape: {df.shape}")
    return df


def build_pipeline():
    """Build the Scikit-learn Pipeline with ColumnTransformer and depth-constrained RandomForest."""
    categorical_features = ["merchant_category", "transaction_type"]
    numerical_features = [
        "amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "day_of_week",
        "is_weekend",
    ]

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
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline, categorical_features, numerical_features


def train_indian_model() -> Tuple[Any, dict]:
    """Train the Indian UPI model using chronological out-of-time partitioning."""
    df = load_or_generate_dataset()

    target = "is_fraud"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Chronological Out-of-Time Partitioning (80% Train, 20% Test)
    # Sort strictly by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)
    n_train = int(n_total * 0.80)

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    cutoff_timestamp = test_df["timestamp"].iloc[0]
    print(f"\nChronological Split (80/20):")
    print(f"  Train: {len(train_df):,} rows (from {train_df['timestamp'].iloc[0][:10]} to {train_df['timestamp'].iloc[-1][:10]})")
    print(f"  Test:  {len(test_df):,} rows (from {test_df['timestamp'].iloc[0][:10]} to {test_df['timestamp'].iloc[-1][:10]})")
    print(f"  Cutoff Timestamp: {cutoff_timestamp}")

    pipeline, cat_cols, num_cols = build_pipeline()
    feature_cols = cat_cols + num_cols

    X_train = train_df[feature_cols]
    y_train = train_df[target].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)

    print(f"\nTraining distribution (Fraud: {y_train.sum()}, Legit: {len(y_train) - y_train.sum()})")
    print(f"Testing distribution  (Fraud: {y_test.sum()}, Legit: {len(y_test) - y_test.sum()})")

    # Fit Model
    print(f"\nTraining depth-constrained RandomForestClassifier (n_estimators=80, max_depth=12)...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Evaluate on held-out test split
    y_pred = pipeline.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n==========================================")
    print(f"HELD-OUT TEST SET EVALUATION (N={len(y_test):,})")
    print(f"==========================================")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # Save Model Artifact (gzip compressed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=("gzip", 3))
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    print(f"\n==========================================")
    print(f"MODEL SERIALIZED SUCCESSFULLY")
    print(f"==========================================")
    print(f"Saved to:  {MODEL_PATH}")
    print(f"File size: {model_size_mb:.2f} MB")

    stats = {
        "training_time_seconds": round(training_time, 4),
        "model_file_size_mb": round(model_size_mb, 4),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "cutoff_timestamp": cutoff_timestamp,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }

    return pipeline, stats


if __name__ == "__main__":
    train_indian_model()
