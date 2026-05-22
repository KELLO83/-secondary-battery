"""Run one ML/DL/foundation model experiment on integrated battery CSVs."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.experiments.runner import run_single_experiment

LOGGER = logging.getLogger(__name__)


MODEL_CHOICES = [
    "lightgbm",
    "catboost",
    "realmlp",
    "tabm",
    "tabr",
    "dcnv2",
    "ft_transformer",
    "tab_transformer",
    "tabnet",
    "tabpfn",
    "tabpfn_latest",
    "tabiclv2",
    "autogluon_mitra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--valid-sample-size", type=int, default=50_000)
    parser.add_argument("--full-data", action="store_true", help="Train on the full integrated Training CSVs.")
    parser.add_argument("--valid-full-data", action="store_true", help="Evaluate on all Validation CSV rows.")
    parser.add_argument("--feature-set", choices=["core_11", "design_15", "chem_22", "chem_derived"], default="core_11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--device", default=None, help="Optional device for neural/foundation models, e.g. cuda or cpu.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=None, help="Optional file path for persistent training logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args)
    sample_size = None if args.full_data else args.sample_size
    valid_sample_size = None if args.valid_full_data else args.valid_sample_size
    model_params = _build_model_params(args)
    LOGGER.info(
        "Starting single experiment: model=%s feature_set=%s sample_size=%s valid_sample_size=%s params=%s",
        args.model,
        args.feature_set,
        sample_size or "full",
        valid_sample_size or "full",
        model_params,
    )
    result = run_single_experiment(
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
        f"valid_mape={result['valid_mape']:.4f}, "
        f"valid_mae={result['valid_mae']:.4f}, "
        f"valid_rmse={result['valid_rmse']:.4f}"
    )


def _build_model_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {}
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
        params.setdefault("task_type", "GPU")
        params.setdefault("devices", "0")
        params.setdefault("gpu_ram_part", 0.90)
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
