"""TabICLv2 wrapper using the official pretrained TabICLRegressor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src.models.base import BaseModel


class TabICLv2Model(BaseModel):
    name = "tabiclv2"
    family = "foundation"

    def __init__(self, feature_set: str = "discharge_summary", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "checkpoint_version": "tabicl-regressor-v2-20260212.ckpt",
            "allow_auto_download": True,
            "kv_cache": False,
            "random_state": 42,
            "verbose": True,
            **(params or {}),
        }
        super().__init__(
            {
                "feature_set": feature_set,
                "params": params,
                "training_mode": "in_context",
                "pretrained": True,
                "checkpoint": params["checkpoint_version"],
                "weight_source": "official_tabicl",
                "access_mode": "local_auto_download",
                "license_checked": False,
            }
        )
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise RuntimeError("TabICLv2 requires the official tabicl package. Install with: pip install tabicl") from exc
        self.model = TabICLRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)

