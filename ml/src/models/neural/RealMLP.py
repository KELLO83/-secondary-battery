"""RealMLP wrapper using the official pytabkit sklearn interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel


class RealMLPModel(BaseModel):
    name = "realmlp"
    family = "neural"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "random_state": 42,
            "n_threads": 14,
            "verbosity": 1,
            "n_epochs": 100,
            "batch_size": 4096,
            **(params or {}),
        }
        if "max_epochs" in params:
            params["n_epochs"] = params.pop("max_epochs")
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from pytabkit import RealMLP_TD_Regressor
        except ImportError as exc:
            raise RuntimeError(
                "RealMLP requires the official pytabkit package. Install with: pip install pytabkit"
            ) from exc

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", RealMLP_TD_Regressor(**params)),
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

