r"""Inspect internal model structures for TabPFN and TabICLv2.

Run from the project root:
    .\.venv314\Scripts\python.exe inspect_foundation_model_structures.py

The foundation estimators expose their torch modules only after their official
packages initialize/load checkpoints. This script tries the least invasive
initialization path first and logs useful fallback metadata when internals are
not available.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml.src.data import feature_registry
from ml.src.models.foundation.TabICLv2 import TabICLv2Model
from ml.src.models.foundation.TabPFN import TabPFNModel

LOGGER = logging.getLogger(__name__)
FEATURE_SET = "inspect_foundation_structures"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    feature_registry.register_feature_set(
        FEATURE_SET,
        feature_columns=["x0", "x1", "x2"],
        categorical_columns=[],
    )

    X, y = _sample_frame()
    inspect_tabpfn()
    #inspect_tabicl(X, y)


def inspect_tabpfn() -> None:
    LOGGER.info("=== TabPFN ===")
    wrapper = TabPFNModel(feature_set=FEATURE_SET, params={"device": "cpu"})
    estimator = wrapper.pipeline.named_steps["model"]
    LOGGER.info("Wrapper config: %s", wrapper.config)
    LOGGER.info("Estimator before init: %s", repr(estimator))

    try:
        # TabPFN v3 exposes model_ after official checkpoint initialization.
        # This avoids fitting data when the package supports config-only init.
        config = estimator.get_inference_config()
        LOGGER.info("TabPFN loaded config: %s", config)
    except Exception:
        LOGGER.exception("TabPFN checkpoint/config initialization failed.")

    _log_internal_model("TabPFN", estimator)


def inspect_tabicl(X: pd.DataFrame, y: pd.Series) -> None:
    LOGGER.info("=== TabICLv2 ===")
    wrapper = TabICLv2Model(
        feature_set=FEATURE_SET,
        params={
            "device": "cpu",
            "allow_auto_download": False,
            "verbose": False,
        },
    )
    estimator = wrapper.model
    LOGGER.info("Wrapper config: %s", wrapper.config)
    LOGGER.info("Estimator before fit/load: %s", repr(estimator))

    try:
        # TabICL creates estimator.model_ while loading its checkpoint in fit().
        # This uses a tiny local sample only to trigger the official load path.
        estimator.fit(X, y)
    except Exception:
        LOGGER.exception("TabICLv2 checkpoint load/fit failed.")

    _log_internal_model("TabICLv2", estimator)


def _sample_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(16, 3)),
        columns=["x0", "x1", "x2"],
    )
    y = pd.Series(X["x0"] * 0.7 - X["x1"] * 0.2 + rng.normal(scale=0.05, size=len(X)))
    return X, y


def _log_internal_model(label: str, estimator: Any) -> None:
    LOGGER.info("%s estimator attrs: %s", label, sorted(vars(estimator).keys()))
    internal = _find_internal_torch_model(estimator)
    if internal is None:
        LOGGER.info("%s internal torch model was not found.", label)
        return

    path, module = internal
    LOGGER.info("%s internal torch model path: %s", label, path)
    _log_torchinfo_summary(label, module)
    LOGGER.info("%s parameter counts: %s", label, _parameter_counts(module))


def _log_torchinfo_summary(label: str, module: Any) -> None:
    try:
        from torchinfo import summary
    except ImportError:
        LOGGER.warning("torchinfo is not installed; install it with: pip install torchinfo")
        return

    try:
        LOGGER.info(
            "%s torchinfo summary:\n%s",
            label,
            summary(
                module,
                depth=3,
                col_names=("num_params", "trainable"),
                verbose=0,
            ),
        )
    except Exception:
        LOGGER.exception("%s torchinfo summary failed.", label)


def _find_internal_torch_model(estimator: Any) -> tuple[str, Any] | None:
    try:
        import torch
    except ImportError:
        LOGGER.info("torch is not installed, cannot type-check internal modules.")
        return None

    candidates = [
        "model_",
        "model",
        "network_",
        "network",
        "module_",
        "module",
    ]
    for name in candidates:
        try:
            value = getattr(estimator, name)
        except Exception:
            continue
        if isinstance(value, torch.nn.Module):
            return name, value

    executor = getattr(estimator, "executor_", None)
    if executor is not None:
        for name in candidates:
            try:
                value = getattr(executor, name)
            except Exception:
                continue
            if isinstance(value, torch.nn.Module):
                return f"executor_.{name}", value
    return None


def _parameter_counts(module: Any) -> dict[str, int]:
    params = list(module.parameters())
    total = sum(param.numel() for param in params)
    trainable = sum(param.numel() for param in params if param.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


if __name__ == "__main__":
    main()
