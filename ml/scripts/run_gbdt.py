"""Run GBDT baselines on NASA cycle-level data."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.experiments.runner import run_sklearn_baseline

LOGGER = logging.getLogger(__name__)
FEATURE_SET_CHOICES = ["cycle_basic", "discharge_summary", "discharge_health"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--valid-sample-size", type=int, default=50_000)
    parser.add_argument("--full-data", action="store_true", help="Train on all NASA train split rows.")
    parser.add_argument(
        "--valid-full-data",
        action="store_true",
        help="Evaluate on all NASA validation split rows instead of a validation sample.",
    )
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="discharge_summary")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=None,
        help="CatBoost task type. Defaults to GPU for catboost and is unsupported for lightgbm.",
    )
    parser.add_argument("--devices", default="0", help="CatBoost GPU device id string, for example '0' or '0:1'.")
    parser.add_argument(
        "--gpu-ram-part",
        type=float,
        default=0.90,
        help="CatBoost GPU RAM fraction for GPU training. Defaults to project policy 0.90.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--log-file", type=Path, default=None, help="Optional file path for persistent training logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args)
    sample_size = None if args.full_data else args.sample_size
    valid_sample_size = None if args.valid_full_data else args.valid_sample_size
    model_params = _build_model_params(args)
    LOGGER.info(
        "Starting GBDT baseline: model=%s feature_set=%s sample_size=%s valid_sample_size=%s params=%s",
        args.model,
        args.feature_set,
        sample_size or "full",
        valid_sample_size or "full",
        model_params,
    )
    result = run_sklearn_baseline(
        model_name=args.model,
        sample_size=sample_size,
        valid_sample_size=valid_sample_size,
        feature_set=args.feature_set,
        seed=args.seed,
        output_path=args.output,
        model_params=model_params,
    )
    print(
        f"{result['experiment_id']}: "
        f"valid_mae={result['valid_mae']:.4f}, "
        f"valid_rmse={result['valid_rmse']:.4f}, "
        f"valid_wape={result['valid_wape']:.4f}"
    )


def _build_model_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {}
    if args.model == "catboost":
        task_type = args.task_type or "GPU"
        params["task_type"] = task_type
        if task_type == "GPU":
            params["devices"] = args.devices
            params["gpu_ram_part"] = args.gpu_ram_part
        return params
    if args.task_type is not None:
        raise ValueError("--task-type is currently supported for catboost baseline only")
    return params


def _setup_logging(args: argparse.Namespace) -> None:
    log_file = args.log_file
    if log_file is None:
        run_scope = "full" if args.full_data else f"sample{args.sample_size}"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("results/logs") / f"{args.model}_{args.feature_set}_{run_scope}_seed{args.seed}_{stamp}.log"
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
