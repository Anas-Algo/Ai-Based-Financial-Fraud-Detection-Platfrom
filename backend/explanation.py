"""
FraudLens Prediction Explanation Engine
Provides deterministic, per-transaction decision-path feature attributions
for the fitted Scikit-learn RandomForestClassifier pipelines (both legacy US
and new Indian UPI models).

Methodology & Mathematical Safeguards:
-------------------------------------
1. Baseline Definition:
   The baseline (prior) is defined as the ensemble's expected fraud probability
   at the root nodes of all decision trees:
       P_base = (1 / M) * sum_{m=1}^M P^{(m)}(root)
   When trained with class_weight='balanced_subsample', P_base reflects
   the balanced root prior across all trees.

2. Exact Ensemble Additivity:
   For any sample traversing tree m from root n_0 to leaf n_k, the leaf probability
   equals the root probability plus the telescoping sum of probability deltas
   at each split node:
       P^{(m)}(leaf) - P^{(m)}(root) = sum_{j=0}^{k-1} [P^{(m)}(n_{j+1}) - P^{(m)}(n_j)]
   Grouping node transitions by split feature f gives the local feature contribution C_f^{(m)}.
   Averaging across all M trees yields:
       P_fraud(x) = P_base + sum_f C_f(x)
   This decomposition matches model.predict_proba down to floating-point precision (< 1e-12).

3. Categorical Aggregation:
   One-hot encoded dummy indicator columns for merchant category and transaction type
   are aggregated into single coherent attributions.

4. Imputed Feature Isolation:
   Uncollected features in legacy models ('gender', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long')
   are isolated into 'imputed_context_contribution' and are NEVER presented as user-provided risk factors.

5. Thresholding & Neutral Wording:
   Only active inputs with absolute contributions >= threshold (|C| >= 0.01)
   are surfaced as factors. Neutral wording is enforced; attributions describe internal
   model decisions and do not claim causal or bank-verified truth.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


# Feature groupings in FraudLens
ACTIVE_USER_FEATURES = {
    "amt",
    "amount",
    "category",
    "merchant_category",
    "transaction_type",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "day_of_week",
    "is_weekend",
}

UNCOLLECTED_CONTEXT_FEATURES = {
    "gender",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
}

DEFAULT_CONTRIBUTION_THRESHOLD = 0.01  # 1.0 percentage point


class DecisionPathExplainer:
    """
    Computes exact, deterministic decision-path feature attributions
    for FraudLens RandomForestClassifier pipelines.
    Supports both legacy US credit card and Indian UPI models.
    """

    def __init__(self, pipeline: Any):
        if not hasattr(pipeline, "named_steps"):
            raise ValueError("Expected a fitted scikit-learn Pipeline with 'named_steps'.")

        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps.get("preprocessor")
        self.classifier = pipeline.named_steps.get("classifier")

        if self.preprocessor is None or self.classifier is None:
            raise ValueError("Pipeline must contain 'preprocessor' and 'classifier' steps.")

        if not hasattr(self.classifier, "estimators_"):
            raise ValueError("Classifier must be a fitted ensemble with 'estimators_'.")

        # Determine fraud class index
        if hasattr(self.classifier, "classes_"):
            classes = list(self.classifier.classes_)
            if 1 in classes:
                self.fraud_class_idx = classes.index(1)
            else:
                self.fraud_class_idx = 1
        else:
            self.fraud_class_idx = 1

        # Extract preprocessed feature names
        if hasattr(self.preprocessor, "get_feature_names_out"):
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        else:
            raise ValueError("Preprocessor does not support get_feature_names_out().")

        self.n_features = len(self.feature_names)
        self.n_estimators = len(self.classifier.estimators_)

        # Identify column indices for categorical aggregations
        self.category_indices: List[int] = [
            i for i, f in enumerate(self.feature_names)
            if f.startswith("categorical__category_") or f.startswith("categorical__merchant_category_")
        ]
        self.type_indices: List[int] = [
            i for i, f in enumerate(self.feature_names)
            if f.startswith("categorical__transaction_type_")
        ]
        self.gender_indices: List[int] = [
            i for i, f in enumerate(self.feature_names)
            if f.startswith("categorical__gender_")
        ]

        # Numeric feature indices
        self.amount_idx: Optional[int] = self._find_first_feature_index(["numerical__amt", "numerical__amount"])
        self.hour_idx: Optional[int] = self._find_first_feature_index(["numerical__transaction_hour"])
        self.day_idx: Optional[int] = self._find_first_feature_index(["numerical__transaction_day"])
        self.month_idx: Optional[int] = self._find_first_feature_index(["numerical__transaction_month"])
        self.weekday_idx: Optional[int] = self._find_first_feature_index(["numerical__day_of_week"])
        self.weekend_idx: Optional[int] = self._find_first_feature_index(["numerical__is_weekend"])

        # Uncollected feature columns in preprocessed space
        self.imputed_feature_indices: List[int] = self.gender_indices + [
            self.feature_names.index(f"numerical__{col}")
            for col in ["lat", "long", "city_pop", "merch_lat", "merch_long"]
            if f"numerical__{col}" in self.feature_names
        ]

        # Cache tree node probability tables and root priors for fast, zero-allocation traversal
        self.cached_tree_data: List[Tuple[Any, np.ndarray, float]] = []
        for est in self.classifier.estimators_:
            tree = est.tree_
            counts = tree.value[:, 0, :]
            # Normalize to class probabilities
            probs = counts[:, self.fraud_class_idx] / counts.sum(axis=1)
            root_prob = float(probs[0])
            self.cached_tree_data.append((tree, probs, root_prob))

        # Base prior: average fraud probability at tree root nodes
        self.base_probability = float(np.mean([item[2] for item in self.cached_tree_data]))

    def _find_first_feature_index(self, names: List[str]) -> Optional[int]:
        for name in names:
            if name in self.feature_names:
                return self.feature_names.index(name)
        return None

    def _find_feature_index(self, name: str) -> int:
        idx = self._find_first_feature_index([name])
        if idx is not None:
            return idx
        raise ValueError(f"Feature '{name}' not found in preprocessor feature names.")

    def explain(
        self,
        input_df: pd.DataFrame,
        threshold: float = DEFAULT_CONTRIBUTION_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Explain a single transaction DataFrame.

        Parameters
        ----------
        input_df : pd.DataFrame
            1-row DataFrame containing the raw transaction features expected by the pipeline.
        threshold : float
            Minimum absolute probability contribution (|C| >= threshold) required
            to present an active feature as a significant model driver.

        Returns
        -------
        Dict[str, Any]
            Deterministic explanation dictionary detailing base probability,
            reconstructed probability, active factors, and isolated imputed baseline.
        """
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("input_df must be a pandas DataFrame.")
        if len(input_df) != 1:
            raise ValueError(f"explain() requires exactly 1 row, got {len(input_df)} rows.")

        row = input_df.iloc[0]

        # 1. Transform row through the preprocessor
        X_t = self.preprocessor.transform(input_df).astype(np.float32)

        # 2. Compute model predicted probability
        predicted_probs = self.classifier.predict_proba(X_t)[0]
        predicted_fraud_prob = float(predicted_probs[self.fraud_class_idx])

        # 3. Traverse decision paths and accumulate feature contributions
        feat_contribs = np.zeros(self.n_features, dtype=np.float64)
        for tree, fraud_probs, _ in self.cached_tree_data:
            path = tree.decision_path(X_t).indices
            for step in range(len(path) - 1):
                curr_n = path[step]
                next_n = path[step + 1]
                split_f = tree.feature[curr_n]
                feat_contribs[split_f] += (fraud_probs[next_n] - fraud_probs[curr_n])

        feat_contribs /= self.n_estimators

        # 4. Strict mathematical additivity verification
        reconstructed_prob = self.base_probability + float(np.sum(feat_contribs))
        reconstruction_error = abs(predicted_fraud_prob - reconstructed_prob)

        if reconstruction_error > 1e-7:
            raise ValueError(
                f"Prediction explanation additivity check failed. "
                f"Reconstructed={reconstructed_prob:.8f}, Expected={predicted_fraud_prob:.8f}, "
                f"Error={reconstruction_error:.2e} exceeds tolerance 1e-7."
            )

        # 5. Aggregate one-hot category dummies into single Merchant Category attribution
        category_contrib = float(np.sum(feat_contribs[self.category_indices])) if self.category_indices else 0.0
        type_contrib = float(np.sum(feat_contribs[self.type_indices])) if self.type_indices else 0.0

        # Active input contributions
        amount_contrib = float(feat_contribs[self.amount_idx]) if self.amount_idx is not None else 0.0
        hour_contrib = float(feat_contribs[self.hour_idx]) if self.hour_idx is not None else 0.0
        day_contrib = float(feat_contribs[self.day_idx]) if self.day_idx is not None else 0.0
        month_contrib = float(feat_contribs[self.month_idx]) if self.month_idx is not None else 0.0
        weekday_contrib = float(feat_contribs[self.weekday_idx]) if self.weekday_idx is not None else 0.0
        weekend_contrib = float(feat_contribs[self.weekend_idx]) if self.weekend_idx is not None else 0.0

        # 6. Isolate imputed features (uncollected context)
        imputed_features_detected: List[str] = [
            col for col in UNCOLLECTED_CONTEXT_FEATURES
            if col not in input_df.columns or pd.isna(row.get(col))
        ]
        imputed_contrib = float(np.sum(feat_contribs[self.imputed_feature_indices])) if self.imputed_feature_indices else 0.0

        # Effective baseline after static imputed training defaults are applied
        effective_baseline = self.base_probability + imputed_contrib

        # 7. Collect and threshold active candidate factors
        # Currency formatting: USD uses $, INR uses Rs / ₹
        is_inr = ("currency" in row and str(row["currency"]).upper() == "INR") or ("amount" in row and "amt" not in row)
        curr_symbol = "Rs " if is_inr else "$"

        raw_amt = float(row["amount"]) if "amount" in row and not pd.isna(row["amount"]) else (
            float(row["amt"]) if "amt" in row and not pd.isna(row["amt"]) else 0.0
        )
        raw_cat = str(row["merchant_category"]) if "merchant_category" in row and not pd.isna(row["merchant_category"]) else (
            str(row["category"]) if "category" in row and not pd.isna(row["category"]) else "other"
        )
        raw_type = str(row["transaction_type"]) if "transaction_type" in row and not pd.isna(row["transaction_type"]) else None
        raw_hour = int(row["transaction_hour"]) if "transaction_hour" in row and not pd.isna(row["transaction_hour"]) else 0
        raw_day = int(row["transaction_day"]) if "transaction_day" in row and not pd.isna(row["transaction_day"]) else 1
        raw_month = int(row["transaction_month"]) if "transaction_month" in row and not pd.isna(row["transaction_month"]) else 1
        raw_weekday = int(row["day_of_week"]) if "day_of_week" in row and not pd.isna(row["day_of_week"]) else None
        raw_weekend = int(row["is_weekend"]) if "is_weekend" in row and not pd.isna(row["is_weekend"]) else None

        candidates = [
            (
                "Amount",
                raw_amt,
                amount_contrib,
                (
                    f"Transaction amount ({curr_symbol}{raw_amt:.2f}) contributed "
                    f"{amount_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ),
            (
                "Merchant Category",
                raw_cat,
                category_contrib,
                (
                    f"Merchant category '{raw_cat}' contributed "
                    f"{category_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ),
            (
                "Transaction Hour",
                raw_hour,
                hour_contrib,
                (
                    f"Transaction hour ({raw_hour:02d}:00) contributed "
                    f"{hour_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ),
            (
                "Transaction Day",
                raw_day,
                day_contrib,
                (
                    f"Transaction day ({raw_day}) contributed "
                    f"{day_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ),
            (
                "Transaction Month",
                raw_month,
                month_contrib,
                (
                    f"Transaction month ({raw_month}) contributed "
                    f"{month_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ),
        ]

        if raw_type is not None and self.type_indices:
            candidates.append((
                "Transaction Type",
                raw_type,
                type_contrib,
                (
                    f"Transaction type '{raw_type}' contributed "
                    f"{type_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ))

        if raw_weekday is not None and self.weekday_idx is not None:
            candidates.append((
                "Day of Week",
                raw_weekday,
                weekday_contrib,
                (
                    f"Day of week ({raw_weekday}) contributed "
                    f"{weekday_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ))

        if raw_weekend is not None and self.weekend_idx is not None:
            candidates.append((
                "Weekend Indicator",
                raw_weekend,
                weekend_contrib,
                (
                    f"Weekend indicator ({raw_weekend}) contributed "
                    f"{weekend_contrib * 100:+.1f}% to model fraud probability relative to baseline."
                ),
            ))

        active_factors: List[Dict[str, Any]] = []
        for feat_name, val, contrib, desc in candidates:
            if abs(contrib) >= threshold:
                active_factors.append({
                    "feature": feat_name,
                    "value": val,
                    "contribution": round(contrib, 4),
                    "impact_percentage": f"{contrib * 100:+.2f}%",
                    "direction": "increases_risk" if contrib > 0 else "decreases_risk",
                    "description": desc,
                })

        # Sort factors by absolute impact descending
        active_factors.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        return {
            "attribution_method": "Decision Tree Path Decomposition (Saabas)",
            "target_class": "Fraudulent",
            "base_probability": round(self.base_probability, 4),
            "predicted_probability": round(predicted_fraud_prob, 4),
            "imputed_context_contribution": round(imputed_contrib, 4),
            "effective_baseline_probability": round(effective_baseline, 4),
            "threshold": threshold,
            "factors": active_factors,
            "imputed_features_isolated": sorted(list(UNCOLLECTED_CONTEXT_FEATURES)),
            "reconstruction_error": round(reconstruction_error, 12),
            "disclaimer": (
                "Model attributions quantify how individual feature values shifted the "
                "Random Forest prediction relative to the ensemble training prior. "
                "They reflect associative patterns within the fitted decision trees, "
                "not external causal evidence or bank verification."
            ),
        }
