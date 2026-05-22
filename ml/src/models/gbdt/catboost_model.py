"""CatBoost regression wrapper for the regular Python 3.14 environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src import schema
from ml.src.models.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"
    family = "gbdt"

    def __init__(self, feature_set: str = "discharge_summary", params: dict[str, Any] | None = None) -> None:
        super().__init__({"feature_set": feature_set, "params": params or {}})
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError(
                "catboost is not installed. Run CatBoost baseline in .venv314 with the official wheel; "
                "do not source-build CatBoost for .venv314t."
            ) from exc

        default_params: dict[str, Any] = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 10,
            "random_seed": 42,
            "thread_count": 14,
            "allow_writing_files": False,
            "verbose": 100,
        }
        default_params.update(params or {})
        self.config["params"] = default_params
        self.feature_set = feature_set
        self.categorical_columns = schema.get_categorical_columns(feature_set)
        self.numeric_columns = schema.get_numeric_columns(feature_set)
        self.model = CatBoostRegressor(**default_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        train = self._prepare_frame(X_train)
        fit_kwargs: dict[str, Any] = {"cat_features": self.categorical_columns}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = (self._prepare_frame(X_valid), y_valid)
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = 50
        self.model.fit(train, y_train, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._prepare_frame(X)), dtype=float)

    def _prepare_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
        return frame

