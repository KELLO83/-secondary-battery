"""Regression metrics used by all model experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    mape: float
    mae: float
    rmse: float
    n_rows: int


def mean_absolute_percentage_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    epsilon: float = 1e-8,
) -> float:
    """Calculate MAPE as a percent with epsilon protection near zero."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denom)) * 100.0)


def mean_absolute_error(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true - pred)))


def root_mean_squared_error(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean(np.square(true - pred))))


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> RegressionMetrics:
    """Return the standard regression metrics for this project."""
    true = np.asarray(y_true, dtype=float)
    return RegressionMetrics(
        mape=mean_absolute_percentage_error(true, y_pred),
        mae=mean_absolute_error(true, y_pred),
        rmse=root_mean_squared_error(true, y_pred),
        n_rows=int(true.shape[0]),
    )


def metrics_by_group(
    y_true: pd.Series,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> dict[str, RegressionMetrics]:
    """Calculate metrics for each source family or other grouping."""
    frame = pd.DataFrame({"y_true": y_true.to_numpy(), "y_pred": y_pred, "group": groups.to_numpy()})
    return {
        str(group): regression_metrics(part["y_true"], part["y_pred"])
        for group, part in frame.groupby("group", dropna=False)
    }
