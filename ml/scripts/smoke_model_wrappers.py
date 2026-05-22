"""Smoke-test model wrappers on synthetic NASA cycle-level data."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src import schema
from ml.src.models.registry import create_model


DEFAULT_MODELS = [
    "dummy_mean",
    "ridge",
    "lightgbm",
    "catboost",
    "realmlp",
    "tabm",
    "tabr",
    "dcnv2",
    "node",
    "ft_transformer",
    "tab_transformer",
    "tabnet",
    "tabiclv2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--feature-set", choices=["cycle_basic", "discharge_summary"], default="discharge_summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    X, y = make_dummy(args.rows)
    for model_name in args.models:
        params = smoke_params(model_name)
        try:
            model = create_model(model_name, feature_set=args.feature_set, params=params)
            model.fit(X.iloc[:24], y.iloc[:24], X.iloc[24:], y.iloc[24:])
            pred = np.asarray(model.predict(X.iloc[24:28]), dtype=float)
            ok = pred.shape == (4,) and np.isfinite(pred).all()
            status = "PASS" if ok else "FAIL"
            detail = f"pred_shape={pred.shape}"
        except Exception as exc:  # noqa: BLE001 - smoke script should report every wrapper failure.
            status = "FAIL"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=1)
        print(f"{model_name}\t{status}\t{detail}")


def make_dummy(n_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for column in schema.get_numeric_columns("discharge_summary"):
        if column == "cycle_index":
            data[column] = np.arange(1, n_rows + 1)
        elif column == "test_id":
            data[column] = np.arange(n_rows)
        elif column == "ambient_temperature":
            data[column] = rng.choice([4.0, 24.0, 43.0], size=n_rows)
        elif column == "sample_count":
            data[column] = rng.integers(400, 1800, size=n_rows)
        elif column == "duration_sec":
            data[column] = rng.normal(3200.0, 300.0, size=n_rows)
        else:
            data[column] = rng.normal(1.0, 0.2, size=n_rows)
    data["battery_id"] = [f"B{idx % 4:04d}" for idx in range(n_rows)]
    X = pd.DataFrame(data)
    y = pd.Series(
        1.8
        - 0.004 * X["cycle_index"].astype(float)
        + 0.00005 * X["duration_sec"].astype(float)
        + rng.normal(0.0, 0.03, size=n_rows),
        name=schema.TARGET_COLUMN,
    )
    return X, y


def smoke_params(model_name: str) -> dict[str, object]:
    params: dict[str, object] = {
        "dummy_mean": {},
        "ridge": {},
        "lightgbm": {"n_estimators": 3, "num_leaves": 7, "min_child_samples": 2, "n_jobs": 2},
        "catboost": {"iterations": 3, "task_type": "CPU", "thread_count": 2, "verbose": False},
        "realmlp": {"device": "cpu", "n_epochs": 1, "batch_size": 16, "verbosity": 0, "n_threads": 2},
        "tabm": {"device": "cpu", "n_epochs": 1, "batch_size": 16, "verbosity": 0, "n_threads": 2},
        "tabr": {
            "device": "cpu",
            "n_epochs": 1,
            "batch_size": 16,
            "eval_batch_size": 16,
            "context_size": 4,
            "candidate_encoding_batch_size": 16,
            "verbosity": 0,
            "n_threads": 2,
        },
        "dcnv2": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 16,
            "verbose": 0,
            "embedding_dim": 4,
            "dnn_hidden_units": (8,),
            "cross_num": 1,
        },
        "node": {
            "device": "cpu",
            "max_epochs": 1,
            "batch_size": 16,
            "num_layers": 1,
            "num_trees": 8,
            "depth": 2,
            "additional_tree_output_dim": 1,
            "progress_bar": "none",
            "early_stopping_patience": 1,
            "num_workers": 0,
        },
        "ft_transformer": {"device": "cpu", "epochs": 1, "batch_size": 16, "dim": 8, "depth": 1, "heads": 2},
        "tab_transformer": {"device": "cpu", "epochs": 1, "batch_size": 16, "dim": 8, "depth": 1, "heads": 2},
        "tabnet": {
            "device_name": "cpu",
            "max_epochs": 1,
            "batch_size": 16,
            "virtual_batch_size": 4,
            "n_d": 4,
            "n_a": 4,
            "n_steps": 2,
        },
        "tabiclv2": {"device": "cpu", "verbose": False, "allow_auto_download": False},
    }.get(model_name, {})
    return params


if __name__ == "__main__":
    main()
