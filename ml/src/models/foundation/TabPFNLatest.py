"""Latest TabPFN wrapper using official pretrained TabPFN weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel
from ml.src.models.foundation.TabPFN import _has_tabpfn_access


class TabPFNLatestModel(BaseModel):
    name = "tabpfn_latest"
    family = "foundation"

    def __init__(self, feature_set: str = "core_11", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "version": "v3",
            "random_state": 42,
            "n_estimators": 8,
            "show_progress_bar": True,
            **(params or {}),
        }
        version = str(params.pop("version"))
        super().__init__(
            {
                "feature_set": feature_set,
                "params": {**params, "version": version},
                "training_mode": "pretrained_inference",
                "pretrained": True,
                "checkpoint": f"tabpfn-{version}",
                "weight_source": "official_tabpfn",
                "access_mode": "local_checkpoint_or_token_or_api",
                "license_checked": _has_tabpfn_access(params),
            }
        )
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise RuntimeError("Latest TabPFN requires the official tabpfn package. Install with: pip install tabpfn") from exc

        model_version = _resolve_model_version(ModelVersion, version)
        model = TabPFNRegressor.create_default_for_version(model_version, **params)
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", model),
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
                "Latest TabPFN pretrained weights require Prior Labs license access before running unattended. "
                "Set TABPFN_TOKEN, place the cached token under ~/.cache/tabpfn/auth_token or ~/.tabpfn/token, "
                "or pass a local model_path/checkpoint."
            )
        self.pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)


def _resolve_model_version(model_version_enum: Any, version: str) -> Any:
    normalized = version.lower().replace("-", "_").replace(".", "_")
    aliases = {
        "latest": "V3",
        "v3": "V3",
        "3": "V3",
        "v2_6": "V2_6",
        "2_6": "V2_6",
        "v2_5": "V2_5",
        "2_5": "V2_5",
        "v2": "V2",
        "2": "V2",
    }
    enum_name = aliases.get(normalized)
    if enum_name is None:
        raise ValueError(f"Unsupported TabPFN version: {version!r}")
    return getattr(model_version_enum, enum_name)
