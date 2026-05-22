"""Preprocessing helpers for sklearn-compatible baselines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src import schema


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    source_family: pd.Series


def filter_target_range(
    df: pd.DataFrame,
    min_target: float = 0.0,
    max_target: float = 1000.0,
) -> pd.DataFrame:
    """Drop rows with missing or out-of-contract target values."""
    target = pd.to_numeric(df[schema.TARGET_COLUMN], errors="coerce")
    mask = target.notna() & (target >= min_target) & (target < max_target)
    return df.loc[mask].copy()


def prepare_xy(df: pd.DataFrame, feature_set: str = "discharge_summary") -> PreparedData:
    """Split dataframe into model features, target, and source-family metadata."""
    df = filter_target_range(df)
    feature_columns = schema.get_feature_columns(feature_set)
    X = df[feature_columns].copy()
    for column in schema.get_numeric_columns(feature_set):
        X[column] = pd.to_numeric(X[column], errors="coerce")
    y = pd.to_numeric(df[schema.TARGET_COLUMN], errors="coerce")
    source_family = df[schema.SOURCE_FAMILY_COLUMN].copy()
    return PreparedData(X=X, y=y, source_family=source_family)


def build_sklearn_preprocessor(feature_set: str = "discharge_summary") -> ColumnTransformer:
    """Build a preprocessing transformer for mixed tabular features."""
    numeric_columns = schema.get_numeric_columns(feature_set)
    categorical_columns = schema.get_categorical_columns(feature_set)

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

