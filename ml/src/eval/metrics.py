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
    wape: float
    smape: float
    filtered_mape: float
    filtered_mape_threshold: float
    filtered_mape_n_rows: int
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


def filtered_mean_absolute_percentage_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    min_abs_target: float = 0.1,
) -> tuple[float, int]:
    """Calculate MAPE only where absolute target is safely away from zero."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(true) > min_abs_target
    if not np.any(mask):
        return float("nan"), 0
    return mean_absolute_percentage_error(true[mask], pred[mask]), int(mask.sum())


def weighted_absolute_percentage_error(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate WAPE as a percent: sum absolute error divided by sum absolute target."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(true - pred)) / denom * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    epsilon: float = 1e-8,
) -> float:
    """Calculate SMAPE as a percent using the average absolute magnitude denominator."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(true) + np.abs(pred)) / 2.0, epsilon)
    return float(np.mean(np.abs(true - pred) / denom) * 100.0)


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
    filtered_mape, filtered_n_rows = filtered_mean_absolute_percentage_error(true, y_pred)
    return RegressionMetrics(
        mape=mean_absolute_percentage_error(true, y_pred),
        mae=mean_absolute_error(true, y_pred),
        rmse=root_mean_squared_error(true, y_pred),
        wape=weighted_absolute_percentage_error(true, y_pred),
        smape=symmetric_mean_absolute_percentage_error(true, y_pred),
        filtered_mape=filtered_mape,
        filtered_mape_threshold=0.1,
        filtered_mape_n_rows=filtered_n_rows,
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
