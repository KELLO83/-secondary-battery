"""Model registry for ML experiments."""

from __future__ import annotations

from typing import Any

from ml.src.models.base import BaseModel
from ml.src.models.foundation.TabICLv2 import TabICLv2Model
from ml.src.models.foundation.TabPFN import TabPFNModel
from ml.src.models.gbdt.catboost_model import CatBoostModel
from ml.src.models.gbdt.lightgbm_model import LightGBMModel
from ml.src.models.neural.DCNV2 import DCNV2Model
from ml.src.models.neural.Node import NodeModel
from ml.src.models.neural.RealMLP import RealMLPModel
from ml.src.models.neural.TabM import TabMModel
from ml.src.models.neural.TabR import TabRModel
from ml.src.models.sklearn_baselines import create_sklearn_baseline
from ml.src.models.transformer.FTTransformer import FTTransformerModel
from ml.src.models.transformer.TabNet import TabNetModel
from ml.src.models.transformer.TabTransformer import TabTransformerModel


def create_model(model_name: str, feature_set: str = "default", params: dict[str, Any] | None = None) -> BaseModel:
    """Create a model wrapper by name."""
    if model_name in {"dummy_mean", "dummy_median", "ridge"}:
        return create_sklearn_baseline(model_name, feature_set=feature_set)
    if model_name == "lightgbm":
        return LightGBMModel(feature_set=feature_set, params=params)
    if model_name == "catboost":
        return CatBoostModel(feature_set=feature_set, params=params)
    if model_name == "realmlp":
        return RealMLPModel(feature_set=feature_set, params=params)
    if model_name == "tabm":
        return TabMModel(feature_set=feature_set, params=params)
    if model_name == "tabr":
        return TabRModel(feature_set=feature_set, params=params)
    if model_name == "dcnv2":
        return DCNV2Model(feature_set=feature_set, params=params)
    if model_name == "node":
        return NodeModel(feature_set=feature_set, params=params)
    if model_name == "ft_transformer":
        return FTTransformerModel(feature_set=feature_set, params=params)
    if model_name == "tab_transformer":
        return TabTransformerModel(feature_set=feature_set, params=params)
    if model_name == "tabnet":
        return TabNetModel(feature_set=feature_set, params=params)
    if model_name == "tabpfn":
        return TabPFNModel(feature_set=feature_set, params=params)
    if model_name == "tabiclv2":
        return TabICLv2Model(feature_set=feature_set, params=params)
    raise ValueError(f"Unsupported model: {model_name}")

