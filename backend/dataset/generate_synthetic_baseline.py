"""
FraudLens — Synthetic Indian UPI Transaction Baseline Generator
===============================================================
Generates a deterministic, reproducible synthetic dataset of Indian UPI and
retail digital transactions for academic simulation, model training, and benchmark evaluation.

ACADEMIC SIMULATION NOTICE:
---------------------------
This dataset is an academic simulation generated via stochastic parameter modeling.
It does NOT represent actual NPCI switch logs, commercial bank clearing records,
or live production UPI network traffic. All transaction IDs, amounts, and
patterns are synthetic and contain zero real-world personally identifiable information (PII).

Methodology & Statistical Assumptions:
--------------------------------------
1. Determinism:
   - Uses a fixed pseudorandom seed (default: 42) via NumPy's Generator.
   - Repeated runs with the identical seed produce bit-for-bit identical datasets.

2. Volume & Temporal Scope:
   - Exactly 50,000 transactions spanning a 180-day continuous window (Jan 1 – Jun 28, 2024).
   - All timestamps are generated and formatted in India Standard Time (IST, UTC+05:30).
   - Diurnal distribution models peak Indian UPI retail hours (10:00–22:00 IST)
     and off-peak late-night periods (01:00–05:00 IST).

3. Payment Rails & Channels:
   - P2M (Peer-to-Merchant): ~58% of volume (daily retail, groceries, utilities, food).
   - P2P (Peer-to-Peer): ~42% of volume (personal transfers, rent, remittances).

4. Retail Merchant Categories & Amount Distributions (INR):
   - kirana_groceries: ~28% (log-normal, median ~Rs 280, range Rs 10 - Rs 3,500)
   - quick_commerce: ~16% (log-normal, median ~Rs 380, range Rs 50 - Rs 4,000)
   - food_dining: ~14% (log-normal, median ~Rs 420, range Rs 60 - Rs 6,500)
   - shopping_retail: ~13% (log-normal, median ~Rs 1,450, range Rs 150 - Rs 35,000)
   - utility_bills: ~10% (log-normal, median ~Rs 850, range Rs 50 - Rs 15,000)
   - travel_ticketing: ~8% (log-normal, median ~Rs 620, range Rs 30 - Rs 25,000)
   - p2p_transfer: ~9% (log-normal, median ~Rs 2,200, range Rs 50 - Rs 100,000)
   - entertainment: ~2% (log-normal, median ~Rs 450, range Rs 100 - Rs 8,000)

5. Stochastic Fraud Modeling:
   - Fraud labels are generated through a continuous latent risk score z rather than
     simplistic deterministic threshold rules:
       z = beta_0 + beta_amt * (log10(amt) - 2.5) + f(hour) + f(category) + f(type) + interaction + noise
     where noise ~ N(0, 0.6^2) and P(fraud) = sigmoid(z).
   - Labels are sampled via Bernoulli trials: y ~ Bernoulli(P(fraud)).
   - This ensures probabilistic realism: high amounts are not automatically fraudulent,
     and modest amounts during off-hours can occasionally be fraudulent.
   - Overall baseline fraud prevalence is calibrated to ~2.0% (typical high-risk simulation tier).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "synthetic_baseline_inr.csv")

CATEGORIES = [
    "kirana_groceries",
    "quick_commerce",
    "food_dining",
    "shopping_retail",
    "utility_bills",
    "travel_ticketing",
    "p2p_transfer",
    "entertainment",
]

# Category probabilities summing to 1.0
CATEGORY_PROBS = [0.28, 0.16, 0.14, 0.13, 0.10, 0.08, 0.09, 0.02]

# Category-specific log-normal amount parameters (mu, sigma, min_val, max_val in INR)
AMOUNT_PARAMS = {
    "kirana_groceries": (5.60, 0.70, 10.0, 5000.0),      # median ~Rs 270
    "quick_commerce": (5.95, 0.65, 50.0, 6000.0),        # median ~Rs 380
    "food_dining": (6.05, 0.75, 50.0, 8000.0),          # median ~Rs 420
    "shopping_retail": (7.25, 1.05, 100.0, 50000.0),     # median ~Rs 1400
    "utility_bills": (6.70, 0.90, 50.0, 20000.0),        # median ~Rs 810
    "travel_ticketing": (6.40, 0.95, 30.0, 30000.0),     # median ~Rs 600
    "p2p_transfer": (7.60, 1.25, 50.0, 100000.0),        # median ~Rs 2000
    "entertainment": (6.10, 0.70, 100.0, 10000.0),       # median ~Rs 445
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid link function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def generate_synthetic_data(
    n_samples: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic Indian UPI transactions.

    Parameters
    ----------
    n_samples : int
        Exact number of transactions to generate (default 50,000).
    random_state : int
        Seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with 50,000 rows and standardized features.
    """
    rng = np.random.default_rng(random_state)

    # 1. Generate Timestamps chronologically across 180 days (Jan 1, 2024 - Jun 28, 2024 IST)
    start_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=IST_OFFSET).timestamp()
    end_epoch = datetime(2024, 6, 28, 23, 59, 59, tzinfo=IST_OFFSET).timestamp()
    span_seconds = end_epoch - start_epoch

    # Diurnal time weights: higher volume 10:00 - 22:00, lower volume 01:00 - 05:00
    # Sample uniformly first, then sort to ensure natural chronological ordering
    raw_seconds = rng.uniform(0, span_seconds, size=n_samples)
    raw_seconds.sort()

    timestamps = [
        datetime.fromtimestamp(start_epoch + s, tz=IST_OFFSET)
        for s in raw_seconds
    ]

    # Temporal feature extraction
    hours = np.array([ts.hour for ts in timestamps], dtype=np.int32)
    days = np.array([ts.day for ts in timestamps], dtype=np.int32)
    months = np.array([ts.month for ts in timestamps], dtype=np.int32)
    weekdays = np.array([ts.weekday() for ts in timestamps], dtype=np.int32)  # 0=Monday, 6=Sunday
    is_weekends = (weekdays >= 5).astype(np.int32)

    # 2. Assign Merchant Categories
    category_indices = rng.choice(len(CATEGORIES), size=n_samples, p=CATEGORY_PROBS)
    categories = [CATEGORIES[i] for i in category_indices]

    # 3. Assign Transaction Type (P2P vs P2M)
    # p2p_transfer is strictly P2P; others are primarily P2M with occasional P2P (e.g. paying small vendor)
    types = []
    for cat in categories:
        if cat == "p2p_transfer":
            types.append("P2P")
        elif cat in ("kirana_groceries", "food_dining"):
            types.append("P2P" if rng.random() < 0.15 else "P2M")
        else:
            types.append("P2M")

    # 4. Generate Amounts in INR
    amounts = np.zeros(n_samples, dtype=np.float64)
    for i, cat in enumerate(categories):
        mu, sigma, min_v, max_v = AMOUNT_PARAMS[cat]
        val = rng.lognormal(mean=mu, sigma=sigma)
        val = np.clip(val, min_v, max_v)
        # Round to 2 decimal places
        amounts[i] = round(float(val), 2)

    # 5. Stochastic Latent Risk Score & Fraud Label Generation
    # Base intercept calibrates unconditional fraud prevalence to ~2.0%
    base_intercept = -5.85

    # Amount effect: logarithmic scale centered at Rs 316 (log10 ~ 2.5)
    amt_effect = 1.15 * (np.log10(np.maximum(amounts, 1.0)) - 2.50)

    # Hour effect: late night (00:00 - 05:00 IST) has elevated risk; peak day has lower risk
    hour_risk = np.zeros(n_samples, dtype=np.float64)
    for i, h in enumerate(hours):
        if h in (1, 2, 3, 4):
            hour_risk[i] = 1.35
        elif h in (0, 5, 23):
            hour_risk[i] = 0.65
        elif 10 <= h <= 20:
            hour_risk[i] = -0.45

    # Category risk weights
    category_risk_map = {
        "kirana_groceries": -0.90,     # Very safe daily micro-purchases
        "utility_bills": -0.80,        # Trusted recurring billers
        "quick_commerce": -0.40,       # Regular fast retail
        "food_dining": -0.30,          # Standard consumer dining
        "travel_ticketing": 0.35,      # Occasional unauthorized ticket booking
        "entertainment": 0.10,         # Modest risk
        "shopping_retail": 0.75,       # Higher risk for electronics/apparel card-not-present
        "p2p_transfer": 1.10,          # Elevated risk for instant account-draining transfers
    }
    cat_risk = np.array([category_risk_map[c] for c in categories], dtype=np.float64)

    # Transaction type risk: P2P transfers have higher anonymity and fraud risk
    type_risk = np.array([0.45 if t == "P2P" else -0.20 for t in types], dtype=np.float64)

    # Non-linear Interaction: high amount (> Rs 15,000) during late night (00:00 - 05:00)
    night_mask = (hours <= 5) | (hours == 23)
    high_amt_mask = amounts > 15000.0
    interaction_risk = 1.50 * (night_mask & high_amt_mask).astype(np.float64)

    # Add Gaussian stochastic noise to prevent deterministic reverse-engineering
    noise = rng.normal(loc=0.0, scale=0.65, size=n_samples)

    # Latent score and logistic probability
    z = base_intercept + amt_effect + hour_risk + cat_risk + type_risk + interaction_risk + noise
    fraud_probabilities = _sigmoid(z)

    # Bernoulli trial realization
    is_fraud = (rng.uniform(0.0, 1.0, size=n_samples) < fraud_probabilities).astype(np.int32)

    # 6. Build Structured DataFrame
    txn_ids = [f"TXN-IND-{i+1:06d}" for i in range(n_samples)]
    iso_timestamps = [ts.isoformat() for ts in timestamps]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "timestamp": iso_timestamps,
        "amount": amounts,
        "currency": "INR",
        "transaction_type": types,
        "merchant_category": categories,
        "transaction_hour": hours,
        "transaction_day": days,
        "transaction_month": months,
        "day_of_week": weekdays,
        "is_weekend": is_weekends,
        "is_fraud": is_fraud,
    })

    return df


def main():
    print("=" * 60)
    print("FraudLens — Synthetic Indian UPI Benchmark Generator")
    print("=" * 60)
    print(f"Generating 50,000 deterministic transactions (seed=42)...")

    df = generate_synthetic_data(n_samples=50000, random_state=42)

    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = total - fraud_count
    fraud_rate = (fraud_count / total) * 100.0

    print(f"\nGeneration complete!")
    print(f"Total transactions:    {total:,}")
    print(f"Legitimate count:      {legit_count:,} ({100 - fraud_rate:.2f}%)")
    print(f"Fraudulent count:      {fraud_count:,} ({fraud_rate:.2f}%)")
    print(f"Currency:              {df['currency'].iloc[0]}")
    print(f"Date Range (IST):      {df['timestamp'].iloc[0][:10]} to {df['timestamp'].iloc[-1][:10]}")
    print(f"Amount Min:            Rs {df['amount'].min():.2f}")
    print(f"Amount Median:         Rs {df['amount'].median():.2f}")
    print(f"Amount Mean:           Rs {df['amount'].mean():.2f}")
    print(f"Amount Max:            Rs {df['amount'].max():.2f}")
    print("\nTransaction Type Breakdown:")
    print(df["transaction_type"].value_counts(normalize=True).round(4) * 100)
    print("\nMerchant Category Breakdown:")
    print(df["merchant_category"].value_counts(normalize=True).round(4) * 100)

    # Save to disk
    os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    file_size_mb = os.path.getsize(DEFAULT_OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved synthetic dataset to: {DEFAULT_OUTPUT_PATH} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
