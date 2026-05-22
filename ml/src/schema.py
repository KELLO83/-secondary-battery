"""Dataset schema constants for the NASA battery cycle-level track."""

from __future__ import annotations

from pathlib import Path

RAW_NASA_ROOT = Path("data/nasa_battery_raw/cleaned_dataset")
NASA_METADATA_FILE = RAW_NASA_ROOT / "metadata.csv"
NASA_SIGNAL_DIR = RAW_NASA_ROOT / "data"
PROCESSED_DATA_DIR = Path("data/processed")
NASA_CYCLE_LEVEL_FILE = PROCESSED_DATA_DIR / "nasa_cycle_level.csv"

TARGET_COLUMN = "capacity"
SOURCE_FAMILY_COLUMN = "battery_id"

NASA_CATEGORICAL_COLUMNS = [
    "battery_id",
]

NASA_BASE_NUMERIC_COLUMNS = [
    "cycle_index",
    "test_id",
    "ambient_temperature",
]

NASA_SIGNAL_NUMERIC_COLUMNS = [
    "sample_count",
    "duration_sec",
    "voltage_measured_first",
    "voltage_measured_last",
    "voltage_measured_min",
    "voltage_measured_max",
    "voltage_measured_mean",
    "voltage_measured_std",
    "current_measured_first",
    "current_measured_last",
    "current_measured_min",
    "current_measured_max",
    "current_measured_mean",
    "current_measured_std",
    "temperature_measured_first",
    "temperature_measured_last",
    "temperature_measured_min",
    "temperature_measured_max",
    "temperature_measured_mean",
    "temperature_measured_std",
    "current_load_first",
    "current_load_last",
    "current_load_min",
    "current_load_max",
    "current_load_mean",
    "current_load_std",
    "voltage_load_first",
    "voltage_load_last",
    "voltage_load_min",
    "voltage_load_max",
    "voltage_load_mean",
    "voltage_load_std",
    "voltage_drop",
    "mean_power_measured",
    "integrated_abs_current",
]

NASA_HEALTH_NUMERIC_COLUMNS = NASA_BASE_NUMERIC_COLUMNS + NASA_SIGNAL_NUMERIC_COLUMNS + [
    "soh",
]

NASA_FEATURE_SETS = {
    "cycle_basic": NASA_BASE_NUMERIC_COLUMNS + NASA_CATEGORICAL_COLUMNS,
    "discharge_summary": NASA_BASE_NUMERIC_COLUMNS + NASA_SIGNAL_NUMERIC_COLUMNS + NASA_CATEGORICAL_COLUMNS,
    "discharge_health": NASA_HEALTH_NUMERIC_COLUMNS + NASA_CATEGORICAL_COLUMNS,
}

SUPPORTED_FEATURE_SETS = tuple(NASA_FEATURE_SETS)
DEFAULT_FEATURE_SET = "discharge_summary"


def get_feature_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return feature columns for a named NASA feature set."""
    try:
        return list(NASA_FEATURE_SETS[feature_set])
    except KeyError as exc:
        raise ValueError(f"Unsupported feature_set: {feature_set!r}") from exc


def get_numeric_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return numeric feature columns for a named NASA feature set."""
    return [column for column in get_feature_columns(feature_set) if column not in NASA_CATEGORICAL_COLUMNS]


def get_categorical_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return categorical feature columns for a named NASA feature set."""
    return [column for column in get_feature_columns(feature_set) if column in NASA_CATEGORICAL_COLUMNS]
