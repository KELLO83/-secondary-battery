from __future__ import annotations

import numpy as np

from ml.src.eval.metrics import regression_metrics


def test_regression_metrics_basic() -> None:
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert round(metrics.mape, 6) == 10.0
    assert metrics.mae == 15.0
    assert round(metrics.rmse, 6) == round((500.0 / 2) ** 0.5, 6)
