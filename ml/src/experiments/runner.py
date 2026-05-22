"""Minimal experiment runner for NASA cycle-level battery data."""

from __future__ import annotations

import time
import sys
import logging
from pathlib import Path

from ml.src.data.loader import sample_integrated_split
from ml.src.data.preprocessing import prepare_xy
from ml.src.eval.metrics import metrics_by_group, regression_metrics
from ml.src.experiments.logger import append_experiment_result
from ml.src.models.registry import create_model

LOGGER = logging.getLogger(__name__)


def run_sklearn_baseline(
    model_name: str,
    sample_size: int | None,
    valid_sample_size: int | None,
    feature_set: str,
    seed: int,
    output_path: Path,
    model_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one model on NASA cycle-level train/validation battery splits."""
    LOGGER.info(
        "Loading data: model=%s feature_set=%s train_sample=%s valid_sample=%s seed=%s",
        model_name,
        feature_set,
        sample_size or "full",
        valid_sample_size or "full",
        seed,
    )
    train_df = sample_integrated_split(
        "train",
        sample_size=sample_size,
        seed=seed,
        feature_set=feature_set,
    )
    valid_df = sample_integrated_split(
        "validation",
        sample_size=valid_sample_size,
        seed=seed,
        feature_set=feature_set,
    )
    LOGGER.info("Loaded raw rows: train=%s validation=%s", len(train_df), len(valid_df))

    LOGGER.info("Preparing feature matrix and target")
    train = prepare_xy(train_df, feature_set=feature_set)
    valid = prepare_xy(valid_df, feature_set=feature_set)
    LOGGER.info(
        "Prepared rows: train=%s validation=%s feature_count=%s",
        len(train.y),
        len(valid.y),
        train.X.shape[1],
    )

    model = create_model(model_name, feature_set=feature_set, params=model_params)
    LOGGER.info(
        "Created model=%s python=%s executable=%s params=%s",
        model_name,
        sys.version.split()[0],
        sys.executable,
        model.config.get("params", {}),
    )

    start_train = time.perf_counter()
    LOGGER.info("Training started")
    model.fit(train.X, train.y, valid.X, valid.y)
    train_time = time.perf_counter() - start_train
    LOGGER.info("Training completed in %.3f sec", train_time)

    start_predict = time.perf_counter()
    LOGGER.info("Prediction started")
    valid_pred = model.predict(valid.X)
    predict_time = time.perf_counter() - start_predict
    LOGGER.info("Prediction completed in %.3f sec", predict_time)

    valid_metrics = regression_metrics(valid.y, valid_pred)
    family_metrics = metrics_by_group(valid.y, valid_pred, valid.source_family)
    LOGGER.info(
        "Validation metrics: rmse=%.6f mae=%.6f wape=%.6f smape=%.6f filtered_mape=%.6f raw_mape=%.6f rows=%s",
        valid_metrics.rmse,
        valid_metrics.mae,
        valid_metrics.wape,
        valid_metrics.smape,
        valid_metrics.filtered_mape,
        valid_metrics.mape,
        valid_metrics.n_rows,
    )
    logged_model_params = model.config.get("params", {})
    cpu_workers = logged_model_params.get("thread_count", logged_model_params.get("n_jobs", ""))
    pretrained = bool(model.config.get("pretrained", False))
    training_mode = str(model.config.get("training_mode", "from_scratch"))

    experiment_id = f"{model_name}_{feature_set}_{sample_size or 'full'}_seed{seed}"
    row = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "model_family": model.family,
        "training_mode": training_mode,
        "pretrained": pretrained,
        "checkpoint": model.config.get("checkpoint", ""),
        "weight_source": model.config.get("weight_source", ""),
        "access_mode": model.config.get("access_mode", ""),
        "license_checked": model.config.get("license_checked", ""),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cpu_workers": cpu_workers,
        "model_params": logged_model_params,
        "feature_set": feature_set,
        "data_size": "nasa_cycle_level",
        "sample_size": sample_size or len(train_df),
        "split_type": "battery_id_group_holdout",
        "split_seed": seed,
        "group_key": "",
        "train_time_sec": round(train_time, 6),
        "predict_time_sec": round(predict_time, 6),
        "valid_mape": valid_metrics.mape,
        "valid_mae": valid_metrics.mae,
        "valid_rmse": valid_metrics.rmse,
        "valid_wape": valid_metrics.wape,
        "valid_smape": valid_metrics.smape,
        "valid_filtered_mape": valid_metrics.filtered_mape,
        "valid_filtered_mape_threshold": valid_metrics.filtered_mape_threshold,
        "valid_filtered_mape_n_rows": valid_metrics.filtered_mape_n_rows,
        "test_mape": "",
        "test_mae": "",
        "test_rmse": "",
        "test_wape": "",
        "test_smape": "",
        "test_filtered_mape": "",
        "test_filtered_mape_threshold": "",
        "test_filtered_mape_n_rows": "",
        "source_family_metrics": family_metrics,
        "notes": (
            "NASA cycle-level model; battery_id group split prevents same-cell train/validation leakage. "
            "Raw MAPE is diagnostic only because zero/near-zero targets inflate percentage error."
        ),
    }
    append_experiment_result(output_path, row)
    LOGGER.info("Experiment result saved: %s", output_path)
    return row


run_single_experiment = run_sklearn_baseline
