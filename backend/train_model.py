import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "fraudTrain.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fraud_detection_pipeline.pkl"
)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Validate dataset
# --------------------------------------------------

target = "is_fraud"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found!"
    )

# Remove rows where the target is missing.
df = df.dropna(subset=[target])

print("\nClass distribution:")
print(df[target].value_counts())


# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------

# Convert transaction time into useful numeric features.
transaction_time = pd.to_datetime(
    df["trans_date_trans_time"],
    errors="coerce"
)

df["transaction_hour"] = transaction_time.dt.hour
df["transaction_day"] = transaction_time.dt.day
df["transaction_month"] = transaction_time.dt.month

# Convert amount to numeric.
df["amt"] = pd.to_numeric(
    df["amt"],
    errors="coerce"
)


# --------------------------------------------------
# 5. Select features
# --------------------------------------------------

# Avoid using personal identifiers such as card number,
# names, street address, and transaction number.

categorical_features = [
    "category",
    "gender",
]

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

features = categorical_features + numerical_features

X = df[features]
y = df[target].astype(int)


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

categorical_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ]
)

numerical_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
        (
            "numerical",
            numerical_transformer,
            numerical_features,
        ),
    ]
)


# --------------------------------------------------
# 7. Machine Learning model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)


# --------------------------------------------------
# 8. Complete pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ]
)


# --------------------------------------------------
# 9. Train-test split
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

print("\nTraining Random Forest model...")
print("This may take some time.")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 11. Evaluate model
# --------------------------------------------------

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# --------------------------------------------------
# 12. Save model
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print("Saved to:", MODEL_PATH)
