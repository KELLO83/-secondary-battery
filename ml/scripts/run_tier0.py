"""Run Tier 0 sanity baselines on NASA cycle-level data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.experiments.runner import run_sklearn_baseline

FEATURE_SET_CHOICES = ["cycle_basic", "discharge_summary", "discharge_health"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy_mean", "dummy_median", "ridge"],
        help="Baseline model names.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--valid-sample-size", type=int, default=None)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="discharge_summary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/experiments.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_name in args.models:
        result = run_sklearn_baseline(
            model_name=model_name,
            sample_size=args.sample_size,
            valid_sample_size=args.valid_sample_size,
            feature_set=args.feature_set,
            seed=args.seed,
            output_path=args.output,
        )
        print(
            f"{result['experiment_id']}: "
            f"valid_mae={result['valid_mae']:.4f}, "
            f"valid_rmse={result['valid_rmse']:.4f}, "
            f"valid_wape={result['valid_wape']:.4f}"
        )


if __name__ == "__main__":
    main()
