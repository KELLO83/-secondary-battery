"""Train one NASA battery cycle-level tabular model experiment.

One invocation runs exactly one model experiment and writes metrics to the
configured results CSV/log file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.experiments.runner import run_single_experiment

LOGGER = logging.getLogger(__name__)

GBDT_MODELS = ["lightgbm", "catboost"]
NEURAL_MODELS = ["realmlp", "tabm", "tabr", "dcnv2", "node"]
TRANSFORMER_MODELS = ["ft_transformer", "tab_transformer", "tabnet"]
FOUNDATION_MODELS = ["tabpfn", "tabpfn_latest", "tabiclv2"]
CEILING_MODELS = ["autogluon_mitra"]
MODEL_CHOICES = GBDT_MODELS + NEURAL_MODELS + TRANSFORMER_MODELS + FOUNDATION_MODELS + CEILING_MODELS
FEATURE_SET_CHOICES = ["cycle_basic", "discharge_summary", "discharge_health"]


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
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lightgbm")
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="discharge_summary")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--valid-sample-size", type=int, default=50_000)
    parser.add_argument("--full-data", action="store_true", help="Train on all NASA train split rows.")
    parser.add_argument("--valid-full-data", action="store_true", help="Evaluate on all NASA validation split rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Optional device for neural/foundation models, e.g. cuda or cpu.")
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
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        print("gbdt:", ", ".join(GBDT_MODELS))
        print("neural:", ", ".join(NEURAL_MODELS))
        print("transformer:", ", ".join(TRANSFORMER_MODELS))
        print("foundation:", ", ".join(FOUNDATION_MODELS))
        print("ceiling:", ", ".join(CEILING_MODELS))
        print("rule: one train.py run = one concrete model only")
        return

    _setup_logging(args)
    _validate_args(args)
    sample_size = None if args.full_data else args.sample_size
    valid_sample_size = None if args.valid_full_data else args.valid_sample_size
    model_params = _build_model_params(args)

    LOGGER.info(
        "Starting experiment: model=%s feature_set=%s sample_size=%s valid_sample_size=%s params=%s",
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
    LOGGER.info(
        "Finished experiment=%s valid_rmse=%.6f valid_mae=%.6f valid_wape=%.6f valid_smape=%.6f",
        result["experiment_id"],
        result["valid_rmse"],
        result["valid_mae"],
        result["valid_wape"],
        result["valid_smape"],
    )
    print(
        f"{result['experiment_id']}: "
        f"valid_rmse={result['valid_rmse']:.4f}, "
        f"valid_mae={result['valid_mae']:.4f}, "
        f"valid_wape={result['valid_wape']:.4f}, "
        f"valid_smape={result['valid_smape']:.4f}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if "," in args.model:
        raise ValueError("One train.py run must train exactly one model.")
    if args.model != "catboost" and args.task_type is not None:
        raise ValueError("--task-type is only supported for catboost.")
    if args.model == "autogluon_mitra" and args.full_data:
        LOGGER.warning("AutoGluon/Mitra full-data runs can be very expensive; prefer sampled ceiling benchmarks first.")


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
