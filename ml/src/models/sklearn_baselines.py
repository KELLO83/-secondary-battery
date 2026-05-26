"""Small sklearn baselines for sanity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel


class SklearnRegressorModel(BaseModel):
    family = "sklearn"

    def __init__(self, name: str, estimator: Any, feature_set: str = "default") -> None:
        super().__init__({"feature_set": feature_set})
        self.name = name
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", estimator),
            ]
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        self.pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)


def create_sklearn_baseline(model_name: str, feature_set: str = "default") -> SklearnRegressorModel:
    """Factory for supported sklearn sanity models."""
    if model_name == "dummy_mean":
        return SklearnRegressorModel(model_name, DummyRegressor(strategy="mean"), feature_set)
    if model_name == "dummy_median":
        return SklearnRegressorModel(model_name, DummyRegressor(strategy="median"), feature_set)
    if model_name == "ridge":
        return SklearnRegressorModel(model_name, Ridge(alpha=1.0, random_state=42), feature_set)
    raise ValueError(f"Unsupported sklearn baseline: {model_name}")

