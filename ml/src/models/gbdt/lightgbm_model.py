"""LightGBM regression wrapper using native categorical features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.src.data import feature_registry
from ml.src.models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"
    family = "gbdt"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        super().__init__({"feature_set": feature_set, "params": params or {}})
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed in this environment") from exc

        default_params: dict[str, Any] = {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 14,
            "random_state": 42,
            "verbosity": -1,
        }
        default_params.update(params or {})
        self.config["params"] = default_params
        self.feature_set = feature_set
        self.categorical_columns = feature_registry.get_categorical_columns(feature_set)
        self.numeric_columns = feature_registry.get_numeric_columns(feature_set)
        self.model = LGBMRegressor(**default_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        train = self._prepare_frame(X_train)
        fit_kwargs: dict[str, Any] = {
            "categorical_feature": self.categorical_columns,
            "callbacks": [_tqdm_callback(int(self.config["params"].get("n_estimators", 0)) or None)],
        }
        if X_valid is not None and y_valid is not None:
            from lightgbm import early_stopping, log_evaluation

            valid = self._prepare_frame(X_valid)
            fit_kwargs["eval_set"] = [(valid, y_valid)]
            fit_kwargs["callbacks"].extend([early_stopping(50), log_evaluation(50)])
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


def _tqdm_callback(total: int | None):
    progress = {"bar": None}

    def _callback(env: Any) -> None:
        if progress["bar"] is None:
            progress["bar"] = tqdm(total=total, desc="lightgbm iterations", unit="iter")
        progress["bar"].update(1)
        if env.iteration + 1 >= env.end_iteration:
            progress["bar"].close()

    _callback.order = 0
    return _callback

