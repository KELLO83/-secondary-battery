"""Domain-derived feature generation for battery tabular experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-8


def add_chem_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe derived features based on chem_22 input columns."""
    result = df.copy()
    voltage_min = _numeric(result, "voltage_range(V)_min")
    voltage_max = _numeric(result, "voltage_range(V)_max")
    li = _numeric(result, "Li_fraction")
    ni = _numeric(result, "Ni_fraction")
    mn = _numeric(result, "Mn_fraction")
    co = _numeric(result, "Co_fraction")
    active = _numeric(result, "active_proportion")
    binder = _numeric(result, "binder_proportion")
    transition_metal = ni + mn + co

    result["voltage_window"] = voltage_max - voltage_min
    result["voltage_mid"] = (voltage_max + voltage_min) / 2.0
    result["Ni_to_Mn"] = _safe_ratio(ni, mn)
    result["Ni_to_Co"] = _safe_ratio(ni, co)
    result["Li_to_TM"] = _safe_ratio(li, transition_metal)
    result["active_to_binder"] = _safe_ratio(active, binder)
    result["total_transition_metal"] = transition_metal
    return result


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator / denominator.where(denominator.abs() > EPSILON, np.nan)
    return values.replace([np.inf, -np.inf], np.nan)
