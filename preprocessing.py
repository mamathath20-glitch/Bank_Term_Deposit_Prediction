"""
preprocessing.py — sklearn ColumnTransformer pipeline for the bank marketing dataset.

Design decisions:
- Numeric features: SimpleImputer (median) + StandardScaler
  pdays is numeric and contains -1 for never-contacted customers.
  StandardScaler normalises the full range including -1.
- Categorical features: SimpleImputer (most_frequent) + OneHotEncoder
  handle_unknown="ignore" ensures unseen categories at inference time don't crash.
  "unknown" string values are kept as explicit category levels so OHE produces
  a dedicated indicator column (e.g. contact_unknown, poutcome_unknown).
- month is one-hot encoded as a nominal categorical feature.
- duration is NOT listed here — it was dropped by data_loader.load_data().
- was_contacted (binary 0/1) is treated as numeric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Feature definitions ───────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",         # original values preserved; -1 means never contacted
    "previous",
    "was_contacted", # engineered: 1 if pdays != -1, else 0
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

TARGET = "y"


def build_preprocessor() -> ColumnTransformer:
    """Build and return a ColumnTransformer for the bank marketing dataset.

    The transformer is NOT fitted here — it is intended to be used inside
    a sklearn Pipeline alongside an estimator:

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", SomeClassifier()),
        ])

    Returns
    -------
    ColumnTransformer
        Ready to be fitted via pipeline.fit(X_train, y_train).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,  # keep all levels including 'unknown'
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop any column not listed above (e.g. target)
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names from a fitted ColumnTransformer."""
    return fitted_preprocessor.get_feature_names_out().tolist()


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the loaded DataFrame into features X and target y."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    X, y = get_X_y(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor)
    print(f"Transformed X shape: {X_transformed.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")
    print("\nFirst 5 feature names:")
    for name in feature_names[:5]:
        print(f"  {name}")
    print("\nLast 5 feature names:")
    for name in feature_names[-5:]:
        print(f"  {name}")
