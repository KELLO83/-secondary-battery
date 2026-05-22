"""TabPFN wrapper using the official pretrained TabPFNRegressor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel


class TabPFNModel(BaseModel):
    name = "tabpfn"
    family = "foundation"

    def __init__(self, feature_set: str = "core_11", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "random_state": 42,
            **(params or {}),
        }
        super().__init__(
            {
                "feature_set": feature_set,
                "params": params,
                "training_mode": "pretrained_inference",
                "pretrained": True,
                "checkpoint": params.get("checkpoint", "default_tabpfn_regressor_checkpoint"),
                "weight_source": "official_tabpfn",
                "access_mode": "local_checkpoint_or_token",
                "license_checked": _has_tabpfn_access(params),
            }
        )
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise RuntimeError("TabPFN requires the official tabpfn package. Install with: pip install tabpfn") from exc

        constructor_params = {key: value for key, value in params.items() if key != "checkpoint"}
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", TabPFNRegressor(**constructor_params)),
            ]
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        if not _has_tabpfn_access(self.config["params"]):
            raise RuntimeError(
                "TabPFN pretrained weights require Prior Labs license access before running unattended. "
                "Set TABPFN_TOKEN, place the cached token under ~/.cache/tabpfn/auth_token or ~/.tabpfn/token, "
                "or pass a local model_path/checkpoint."
            )
        self.pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)


def _has_tabpfn_access(params: dict[str, Any]) -> bool:
    model_path = params.get("model_path") or params.get("checkpoint")
    if model_path and str(model_path) != "auto":
        return True
    if os.environ.get("TABPFN_TOKEN"):
        return True
    token_paths = [
        Path.home() / ".cache" / "tabpfn" / "auth_token",
        Path.home() / ".tabpfn" / "token",
    ]
    return any(path.exists() and path.read_text(encoding="utf-8").strip() for path in token_paths)
