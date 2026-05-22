"""AutoGluon Mitra ceiling benchmark wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src import schema
from ml.src.models.base import BaseModel


class AutoGluonMitraModel(BaseModel):
    name = "autogluon_mitra"
    family = "automl_ceiling"

    def __init__(self, feature_set: str = "discharge_summary", params: dict[str, Any] | None = None) -> None:
        params = {
            "path": "results/autogluon/mitra",
            "time_limit": None,
            "presets": None,
            "verbosity": 2,
            "hyperparameters": {"MITRA": {"fine_tune": False}},
            **(params or {}),
        }
        super().__init__(
            {
                "feature_set": feature_set,
                "params": params,
                "training_mode": "foundation_ceiling",
                "pretrained": True,
                "checkpoint": "autogluon/mitra-regressor",
                "weight_source": "autogluon_mitra",
                "access_mode": "local_auto_download",
                "license_checked": True,
            }
        )
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:
            raise RuntimeError(
                "AutoGluon Mitra requires AutoGluon foundational model extras. "
                "Install in .venv314 with: uv pip install autogluon.tabular[mitra]"
            ) from exc
        self._predictor_cls = TabularPredictor
        self.predictor: Any | None = None
        self.path = Path(str(params["path"]))

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        train_data = X_train.copy()
        train_data[schema.TARGET_COLUMN] = y_train.to_numpy()
        tuning_data = None
        if X_valid is not None and y_valid is not None:
            tuning_data = X_valid.copy()
            tuning_data[schema.TARGET_COLUMN] = y_valid.to_numpy()

        params = self.config["params"]
        self.path.mkdir(parents=True, exist_ok=True)
        self.predictor = self._predictor_cls(
            label=schema.TARGET_COLUMN,
            problem_type="regression",
            path=str(self.path),
            verbosity=int(params.get("verbosity", 2)),
        )
        self.predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=params.get("hyperparameters"),
            presets=params.get("presets"),
            time_limit=params.get("time_limit"),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("AutoGluon Mitra predictor is not fitted")
        return np.asarray(self.predictor.predict(X), dtype=float)

