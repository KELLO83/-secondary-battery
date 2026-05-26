"""Train one generic CSV-backed tabular regression experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LOGGER = logging.getLogger(__name__)

SKLEARN_MODELS = ["dummy_mean", "dummy_median", "ridge"]
GBDT_MODELS = ["lightgbm", "catboost"]
NEURAL_MODELS = ["realmlp", "tabm", "tabr", "dcnv2", "node"]
TRANSFORMER_MODELS = ["ft_transformer", "tab_transformer", "tabnet"]
FOUNDATION_MODELS = ["tabpfn", "tabiclv2"]
MODEL_CHOICES = SKLEARN_MODELS + GBDT_MODELS + NEURAL_MODELS + TRANSFORMER_MODELS + FOUNDATION_MODELS
DEFAULT_CSV = ROOT / "superconductivity" / "openml_44964_superconductivity.csv"
DEFAULT_TARGET = "critical_temp"
DEFAULT_MODEL = "lightgbm"


def parse_key_value(raw: str) -> tuple[str, Any]:
    """Parse KEY=VALUE CLI overrides with basic scalar conversion."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Tabular regression JSON config.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Input CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Regression target column. Default: {DEFAULT_TARGET}")
    parser.add_argument("--features", default="", help="Comma-separated feature columns.")
    parser.add_argument("--exclude", default="", help="Comma-separated columns to exclude from automatic features.")
    parser.add_argument("--categorical", default="", help="Comma-separated categorical feature columns.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/tabular_regression_experiments.csv"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--device", default=None, help="Optional model device, e.g. cuda or cpu.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=None, help="CatBoost task type.")
    parser.add_argument("--devices", default="0", help="CatBoost GPU device id string.")
    parser.add_argument("--gpu-ram-part", type=float, default=0.90)
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Override model config. Example: --model-param n_estimators=200",
    )
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_models:
        print("sklearn:", ", ".join(SKLEARN_MODELS))
        print("gbdt:", ", ".join(GBDT_MODELS))
        print("neural:", ", ".join(NEURAL_MODELS))
        print("transformer:", ", ".join(TRANSFORMER_MODELS))
        print("foundation:", ", ".join(FOUNDATION_MODELS))
        print("rule: one train.py run = one concrete model only")
        return

    _setup_logging(args)
    _validate_args(args)
    result = _run_generic_tabular_mode(args)
    LOGGER.info(
        "Finished experiment=%s valid_rmse=%.6f valid_mae=%.6f valid_wape=%.6f",
        result["experiment_id"],
        result["rmse"],
        result["mae"],
        result["wape"],
    )
    print(
        f"{result['experiment_id']}: "
        f"valid_rmse={result['rmse']:.4f}, "
        f"valid_mae={result['mae']:.4f}, "
        f"valid_wape={result['wape']:.4f}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if "," in args.model:
        raise ValueError("One train.py run must train exactly one model.")
    if args.model != "catboost" and args.task_type is not None:
        raise ValueError("--task-type is only supported for catboost.")
    if args.config is None and (args.csv is None or args.target is None):
        raise ValueError("Provide --config, or provide --csv and --target.")


def _build_model_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = dict(args.model_param)
    if args.device is not None:
        params["device"] = args.device
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
        params["n_epochs"] = args.max_epochs
        if args.model in {"ft_transformer", "tab_transformer", "dcnv2"}:
            params["epochs"] = args.max_epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.model == "catboost":
        task_type = args.task_type or "GPU"
        params["task_type"] = task_type
        if task_type == "GPU":
            params.setdefault("devices", args.devices)
            params.setdefault("gpu_ram_part", args.gpu_ram_part)
    return params


def _run_generic_tabular_mode(args: argparse.Namespace) -> dict[str, Any]:
    module = _load_tabular_regression_module()
    args = module.resolve_config(args)
    params = _build_model_params(args)
    df = module.read_csv(args.csv)
    config = module.infer_columns(
        df=df,
        target=args.target,
        features=module.parse_csv_list(args.features),
        exclude=module.parse_csv_list(args.exclude),
        categorical=module.parse_csv_list(args.categorical),
    )
    frame = module.clean_training_frame(df, config)
    train_df, valid_df = module.train_test_split(frame, test_size=args.test_size, random_state=args.seed)

    X_train = train_df[config.feature_columns].copy()
    y_train = train_df[config.target].copy()
    X_valid = valid_df[config.feature_columns].copy()
    y_valid = valid_df[config.target].copy()

    feature_set_name = module.register_runtime_feature_set(args, config)
    model = module.build_model(
        model_name=args.model,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        params=params,
        feature_set_name=feature_set_name,
    )
    import time
    import numpy as np

    start_train = time.perf_counter()
    model.fit(X_train, y_train, X_valid, y_valid)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    pred = np.asarray(model.predict(X_valid), dtype=float)
    predict_time = time.perf_counter() - start_predict
    metrics = module.regression_metrics(y_valid, pred)
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
        "excluded_columns": module.json.dumps(config.excluded_columns, ensure_ascii=False),
        "feature_columns": module.json.dumps(config.feature_columns, ensure_ascii=False),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_time_sec": round(train_time, 6),
        "predict_time_sec": round(predict_time, 6),
        **metrics,
        "model_params": module.json.dumps(params, ensure_ascii=False),
    }
    module.append_result(args.output, row)
    if args.save_predictions:
        module.save_predictions(args.prediction_dir, experiment_id, valid_df, y_valid, pred)
    return row


def _load_tabular_regression_module():
    path = ROOT / "ml" / "scripts" / "run_tabular_regression.py"
    spec = importlib.util.spec_from_file_location("run_tabular_regression", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generic tabular runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_logging(args: argparse.Namespace) -> None:
    log_file = args.log_file
    if log_file is None:
        log_model = args.model
        log_target = args.target or "config"
        if args.config is not None and args.config.exists():
            import json

            config = json.loads(args.config.read_text(encoding="utf-8-sig"))
            log_model = str(config.get("model", log_model))
            log_target = str(config.get("target", log_target))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("results/logs") / f"{log_model}_{log_target}_seed{args.seed}_{stamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    LOGGER.info("Persistent log file: %s", log_file)


if __name__ == "__main__":
    main()
