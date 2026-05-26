"""Run a generic tabular regression experiment from a CSV file.

This script is dataset-agnostic: pass a CSV path and target column, and it
will train/evaluate a regression model on the remaining usable columns.

Examples:
  python ml/scripts/run_tabular_regression.py --csv "data/my.csv" --target price --model lightgbm
  python ml/scripts/run_tabular_regression.py --config "configs/experiment.json"
  python ml/scripts/run_tabular_regression.py --csv "data/my.csv" --target y --exclude id,date --model tabpfn --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data import feature_registry
from ml.src.models.registry import create_model

DEFAULT_OUTPUT = Path("results/tabular_regression_experiments.csv")
DEFAULT_PREDICTION_DIR = Path("results/predictions")
MODEL_CHOICES = [
    "dummy_mean",
    "dummy_median",
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
    "tabpfn",
    "tabiclv2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON experiment config.")
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV file.")
    parser.add_argument("--target", default=None, help="Regression target column.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lightgbm")
    parser.add_argument("--features", default="", help="Comma-separated feature columns. Defaults to all usable columns.")
    parser.add_argument("--exclude", default="", help="Comma-separated columns to exclude from features.")
    parser.add_argument("--categorical", default="", help="Comma-separated categorical feature columns.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--device", default=None, help="Optional device for TabPFN, e.g. cuda or cpu.")
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Override model config. Example: --model-param n_estimators=100",
    )
    return parser


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    elif lowered in {"none", "null"}:
        parsed = None
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    return key.strip(), parsed


def main() -> None:
    args = build_parser().parse_args()
    args = resolve_config(args)
    params = dict(args.model_param)
    if args.device is not None:
        params["device"] = args.device

    df = read_csv(args.csv)
    config = infer_columns(
        df=df,
        target=args.target,
        features=parse_csv_list(args.features),
        exclude=parse_csv_list(args.exclude),
        categorical=parse_csv_list(args.categorical),
    )
    frame = clean_training_frame(df, config)
    train_df, valid_df = train_test_split(frame, test_size=args.test_size, random_state=args.seed)

    X_train = train_df[config.feature_columns].copy()
    y_train = train_df[config.target].copy()
    X_valid = valid_df[config.feature_columns].copy()
    y_valid = valid_df[config.target].copy()

    feature_set_name = register_runtime_feature_set(args, config)
    model = build_model(
        model_name=args.model,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        params=params,
        feature_set_name=feature_set_name,
    )

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    pred = np.asarray(model.predict(X_valid), dtype=float)
    predict_time = time.perf_counter() - start_predict

    metrics = regression_metrics(y_valid, pred)
    experiment_id = f"{args.csv.stem}_{args.target}_{args.model}_{len(train_df)}_seed{args.seed}"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "csv": str(args.csv),
        "target": args.target,
        "model": args.model,
        "rows": len(frame),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "feature_count": len(config.feature_columns),
        "numeric_count": len(config.numeric_columns),
        "categorical_count": len(config.categorical_columns),
        "excluded_columns": json.dumps(config.excluded_columns, ensure_ascii=False),
        "feature_columns": json.dumps(config.feature_columns, ensure_ascii=False),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_time_sec": round(train_time, 6),
        "predict_time_sec": round(predict_time, 6),
        **metrics,
        "model_params": json.dumps(params, ensure_ascii=False),
    }
    append_result(args.output, row)
    if args.save_predictions:
        save_predictions(args.prediction_dir, experiment_id, valid_df, y_valid, pred)

    print(
        f"{experiment_id}: "
        f"rmse={metrics['rmse']:.6f}, "
        f"mae={metrics['mae']:.6f}, "
        f"wape={metrics['wape']:.6f}, "
        f"features={len(config.feature_columns)}, "
        f"train={len(train_df)}, valid={len(valid_df)}"
    )


class ColumnConfig:
    def __init__(
        self,
        target: str,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
        excluded_columns: list[str],
    ) -> None:
        self.target = target
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.excluded_columns = excluded_columns


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def infer_columns(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    exclude: list[str],
    categorical: list[str],
) -> ColumnConfig:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target!r}")

    auto_exclude = [column for column in df.columns if column.startswith("Unnamed:")]
    excluded = sorted(set(exclude + auto_exclude + [target]))
    missing_exclude = sorted(set(exclude) - set(df.columns))
    if missing_exclude:
        raise ValueError(f"Excluded columns not found: {missing_exclude}")
    missing_categorical = sorted(set(categorical) - set(df.columns))
    if missing_categorical:
        raise ValueError(f"Categorical columns not found: {missing_categorical}")

    if features:
        missing_features = sorted(set(features) - set(df.columns))
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        overlap = sorted(set(features) & set(excluded))
        if overlap:
            raise ValueError(f"Columns cannot be both features and excluded: {overlap}")
        feature_columns = list(features)
    else:
        feature_columns = [column for column in df.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    explicit_categorical = set(categorical)
    categorical_columns = [
        column
        for column in feature_columns
        if column in explicit_categorical or not pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    return ColumnConfig(
        target=target,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_columns=excluded,
    )


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        if args.csv is None or args.target is None:
            raise ValueError("--csv and --target are required unless --config is provided.")
        return args

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in [
        "csv",
        "target",
        "model",
        "features",
        "exclude",
        "categorical",
        "test_size",
        "seed",
        "output",
        "device",
    ]:
        if key not in config:
            continue
        value = config[key]
        if key in {"csv", "output"} and value is not None:
            value = Path(value)
        if key in {"features", "exclude", "categorical"} and isinstance(value, list):
            value = ",".join(value)
        setattr(args, key, value)
    if "model_params" in config:
        args.model_param.extend(config["model_params"].items())
    if args.csv is None or args.target is None:
        raise ValueError("Config must provide csv and target, or CLI must override them.")
    return args


def clean_training_frame(df: pd.DataFrame, config: ColumnConfig) -> pd.DataFrame:
    frame = df[config.feature_columns + [config.target]].copy()
    frame[config.target] = pd.to_numeric(frame[config.target], errors="coerce")
    for column in config.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in config.categorical_columns:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame.dropna(subset=[config.target]).reset_index(drop=True)


def build_model(
    model_name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    feature_set_name: str | None = None,
):
    if feature_set_name is not None:
        if model_name == "tabpfn":
            load_tabpfn_token()
        return create_model(model_name, feature_set=feature_set_name, params=params)

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    from sklearn.linear_model import Ridge

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge())])


def register_runtime_feature_set(args: argparse.Namespace, config: ColumnConfig) -> str:
    feature_set_name = f"csv_{Path(args.csv).stem}_{args.target}_{args.seed}"
    feature_registry.register_feature_set(
        feature_set_name,
        feature_columns=config.feature_columns,
        categorical_columns=config.categorical_columns,
    )
    return feature_set_name


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def load_tabpfn_token() -> None:
    if os.environ.get("TABPFN_TOKEN"):
        return
    token_path = ROOT / ".secrets" / "tabpfn_token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff")
        if token:
            os.environ["TABPFN_TOKEN"] = token


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    abs_error = np.abs(true - y_pred)
    denom = np.maximum(np.abs(true), 1e-8)
    sum_target = np.sum(np.abs(true))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(math.sqrt(np.mean(np.square(true - y_pred)))),
        "mape": float(np.mean(abs_error / denom) * 100.0),
        "wape": float(np.sum(abs_error) / sum_target * 100.0) if sum_target else float("nan"),
    }


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_predictions(
    prediction_dir: Path,
    experiment_id: str,
    valid_df: pd.DataFrame,
    y_valid: pd.Series,
    pred: np.ndarray,
) -> None:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / f"{experiment_id}_predictions.csv"
    frame = valid_df.copy()
    frame["y_true"] = y_valid.to_numpy()
    frame["y_pred"] = pred
    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
    frame.to_csv(output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
