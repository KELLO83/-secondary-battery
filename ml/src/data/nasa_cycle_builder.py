"""Build a cycle-level tabular dataset from NASA discharge CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.src import schema

SIGNAL_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
]


def build_nasa_cycle_level_dataset(
    raw_root: Path = schema.RAW_NASA_ROOT,
    output_path: Path = schema.NASA_CYCLE_LEVEL_FILE,
    force: bool = False,
) -> pd.DataFrame:
    """Create or load the processed NASA discharge cycle-level table."""
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    metadata_path = raw_root / "metadata.csv"
    signal_dir = raw_root / "data"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing NASA metadata file: {metadata_path}")
    if not signal_dir.exists():
        raise FileNotFoundError(f"Missing NASA signal directory: {signal_dir}")

    meta = pd.read_csv(metadata_path)
    discharge = meta.loc[meta["type"].eq("discharge")].copy()
    discharge["capacity"] = pd.to_numeric(discharge["Capacity"], errors="coerce")
    discharge = discharge.dropna(subset=["capacity", "battery_id", "filename"])
    discharge = discharge.sort_values(["battery_id", "test_id", "uid"]).reset_index(drop=True)
    discharge["cycle_index"] = discharge.groupby("battery_id").cumcount() + 1
    first_capacity = discharge.groupby("battery_id")["capacity"].transform("first")
    discharge["soh"] = discharge["capacity"] / first_capacity.replace(0, np.nan)

    rows: list[dict[str, object]] = []
    iterator = tqdm(discharge.itertuples(index=False), total=len(discharge), desc="Building NASA cycles", unit="cycle")
    for row in iterator:
        signal_path = signal_dir / str(row.filename)
        if not signal_path.exists():
            continue
        signal = pd.read_csv(signal_path)
        features = _summarize_discharge_signal(signal)
        features.update(
            {
                "battery_id": row.battery_id,
                "test_id": int(row.test_id),
                "uid": int(row.uid),
                "filename": row.filename,
                "ambient_temperature": float(row.ambient_temperature),
                "cycle_index": int(row.cycle_index),
                "capacity": float(row.capacity),
                "soh": float(row.soh),
            }
        )
        rows.append(features)

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def _summarize_discharge_signal(signal: pd.DataFrame) -> dict[str, float]:
    signal = signal.copy()
    for column in [*SIGNAL_COLUMNS, "Time"]:
        if column in signal.columns:
            signal[column] = pd.to_numeric(signal[column], errors="coerce")

    summary: dict[str, float] = {
        "sample_count": float(len(signal)),
        "duration_sec": _duration(signal),
    }
    for column in SIGNAL_COLUMNS:
        key = _snake(column)
        values = signal[column] if column in signal.columns else pd.Series(dtype=float)
        summary[f"{key}_first"] = _first_valid(values)
        summary[f"{key}_last"] = _last_valid(values)
        summary[f"{key}_min"] = float(values.min(skipna=True)) if len(values) else np.nan
        summary[f"{key}_max"] = float(values.max(skipna=True)) if len(values) else np.nan
        summary[f"{key}_mean"] = float(values.mean(skipna=True)) if len(values) else np.nan
        summary[f"{key}_std"] = float(values.std(skipna=True)) if len(values) else np.nan

    summary["voltage_drop"] = summary["voltage_measured_first"] - summary["voltage_measured_last"]
    if {"Voltage_measured", "Current_measured"}.issubset(signal.columns):
        summary["mean_power_measured"] = float((signal["Voltage_measured"] * signal["Current_measured"]).mean(skipna=True))
    else:
        summary["mean_power_measured"] = np.nan
    if {"Current_measured", "Time"}.issubset(signal.columns):
        summary["integrated_abs_current"] = _integrate_abs_current(signal["Current_measured"], signal["Time"])
    else:
        summary["integrated_abs_current"] = np.nan
    return summary


def _duration(signal: pd.DataFrame) -> float:
    if "Time" not in signal.columns or signal["Time"].dropna().empty:
        return np.nan
    return float(signal["Time"].max(skipna=True) - signal["Time"].min(skipna=True))


def _first_valid(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[0]) if not clean.empty else np.nan


def _last_valid(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _integrate_abs_current(current: pd.Series, time: pd.Series) -> float:
    frame = pd.DataFrame({"current": current.abs(), "time": time}).dropna().sort_values("time")
    if len(frame) < 2:
        return np.nan
    return float(np.trapezoid(frame["current"].to_numpy(), frame["time"].to_numpy()))


def _snake(column: str) -> str:
    return column.strip().lower()
