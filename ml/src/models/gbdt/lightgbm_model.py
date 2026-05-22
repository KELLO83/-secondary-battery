"""LightGBM regression wrapper using native categorical features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src import schema
from ml.src.models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"
    family = "gbdt"

    def __init__(self, feature_set: str = "core_11", params: dict[str, Any] | None = None) -> None:
        super().__init__({"feature_set": feature_set, "params": params or {}})
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed in this environment") from exc

        default_params: dict[str, Any] = {
            "objective": "regression",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 100,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 14,
            "random_state": 42,
            "verbosity": -1,
        }
        default_params.update(params or {})
        self.config["params"] = default_params
        self.feature_set = feature_set
        self.categorical_columns = schema.get_categorical_columns(feature_set)
        self.numeric_columns = schema.get_numeric_columns(feature_set)
        self.model = LGBMRegressor(**default_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        train = self._prepare_frame(X_train)
        fit_kwargs: dict[str, Any] = {"categorical_feature": self.categorical_columns}
        if X_valid is not None and y_valid is not None:
            from lightgbm import early_stopping, log_evaluation

            valid = self._prepare_frame(X_valid)
            fit_kwargs["eval_set"] = [(valid, y_valid)]
            fit_kwargs["callbacks"] = [early_stopping(50), log_evaluation(0)]
        self.model.fit(train, y_train, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._prepare_frame(X)), dtype=float)

    def _prepare_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype("category")
        return frame
