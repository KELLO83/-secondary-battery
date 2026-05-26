"""Preprocessing helpers for generic tabular model wrappers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.src.data import feature_registry


def build_sklearn_preprocessor(feature_set: str) -> ColumnTransformer:
    """Build a preprocessing transformer for mixed tabular features."""
    numeric_columns = feature_registry.get_numeric_columns(feature_set)
    categorical_columns = feature_registry.get_categorical_columns(feature_set)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("stringify", FunctionTransformer(_stringify_categories, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _stringify_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to strings and normalize missing values."""
    return frame.astype("string").fillna("__MISSING__")
