"""FT-Transformer wrapper using the tabular-transformers package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.src import schema
from ml.src.models.base import BaseModel


class FTTransformerModel(BaseModel):
    name = "ft_transformer"
    family = "transformer"

    def __init__(self, feature_set: str = "core_11", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "dim": 32,
            "depth": 4,
            "heads": 4,
            "epochs": 50,
            "batch_size": 2048,
            "learning_rate": 1e-3,
            **(params or {}),
        }
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from fttransformer.model import FTTransformer
            import torch
        except ImportError as exc:
            raise RuntimeError("FT-Transformer requires tabular-transformers and torch.") from exc
        self.feature_set = feature_set
        self.params = params
        self.torch = torch
        self.model_cls = FTTransformer
        self.model: Any | None = None
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None = None, y_valid: pd.Series | None = None) -> None:
        x_cat, x_num = self._fit_transform(X_train)
        categories = [len(self.category_maps[col]) + 1 for col in schema.get_categorical_columns(self.feature_set)]
        self.model = self.model_cls(
            categories=categories,
            num_continuous=len(schema.get_numeric_columns(self.feature_set)),
            dim=int(self.params["dim"]),
            dim_out=1,
            depth=int(self.params["depth"]),
            heads=int(self.params["heads"]),
        ).to(self.params["device"])
        self._train_torch_model(x_cat, x_num, y_train.to_numpy(dtype=np.float32))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FTTransformerModel is not fitted")
        x_cat, x_num = self._transform(X)
        self.model.eval()
        preds = []
        batch_size = int(self.params["batch_size"])
        with self.torch.no_grad():
            for start in range(0, len(X), batch_size):
                out = self.model(
                    self.torch.as_tensor(x_cat[start : start + batch_size], dtype=self.torch.long, device=self.params["device"]),
                    self.torch.as_tensor(x_num[start : start + batch_size], dtype=self.torch.float32, device=self.params["device"]),
                )
                preds.append(out.detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds)

    def _train_torch_model(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> None:
        assert self.model is not None
        dataset = self.torch.utils.data.TensorDataset(
            self.torch.as_tensor(x_cat, dtype=self.torch.long),
            self.torch.as_tensor(x_num, dtype=self.torch.float32),
            self.torch.as_tensor(y.reshape(-1, 1), dtype=self.torch.float32),
        )
        loader = self.torch.utils.data.DataLoader(dataset, batch_size=int(self.params["batch_size"]), shuffle=True)
        optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=float(self.params["learning_rate"]))
        loss_fn = self.torch.nn.MSELoss()
        self.model.train()
        for _ in tqdm(range(int(self.params["epochs"])), desc="ft_transformer epochs"):
            for batch_cat, batch_num, batch_y in loader:
                batch_cat = batch_cat.to(self.params["device"])
                batch_num = batch_num.to(self.params["device"])
                batch_y = batch_y.to(self.params["device"])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(batch_cat, batch_num), batch_y)
                loss.backward()
                optimizer.step()

    def _fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        for col in schema.get_categorical_columns(self.feature_set):
            values = X[col].astype("string").fillna("__MISSING__").astype(str)
            self.category_maps[col] = {value: idx + 1 for idx, value in enumerate(sorted(values.unique()))}
        for col in schema.get_numeric_columns(self.feature_set):
            values = pd.to_numeric(X[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std()) or 1.0
            self.numeric_medians[col] = median
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_cols = schema.get_categorical_columns(self.feature_set)
        num_cols = schema.get_numeric_columns(self.feature_set)
        x_cat = np.column_stack([
            X[col].astype("string").fillna("__MISSING__").astype(str).map(self.category_maps[col]).fillna(0).to_numpy(dtype=np.int64)
            for col in cat_cols
        ])
        x_num = np.column_stack([
            ((pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians[col]) - self.numeric_means[col]) / self.numeric_stds[col]).to_numpy(dtype=np.float32)
            for col in num_cols
        ])
        return x_cat, x_num
