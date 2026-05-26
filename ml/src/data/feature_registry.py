"""Runtime feature registry for generic tabular experiments."""

from __future__ import annotations

DEFAULT_FEATURE_SET = "default"

CUSTOM_FEATURE_SETS: dict[str, list[str]] = {}
CUSTOM_CATEGORICAL_COLUMNS: dict[str, list[str]] = {}


def get_feature_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return feature columns for a registered feature set."""
    try:
        return list(CUSTOM_FEATURE_SETS[feature_set])
    except KeyError as exc:
        raise ValueError(f"Unsupported feature_set: {feature_set!r}") from exc


def get_numeric_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return numeric feature columns for a registered feature set."""
    categorical = set(get_categorical_columns(feature_set))
    return [column for column in get_feature_columns(feature_set) if column not in categorical]


def get_categorical_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Return categorical feature columns for a registered feature set."""
    return list(CUSTOM_CATEGORICAL_COLUMNS.get(feature_set, []))


def register_feature_set(
    name: str,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
) -> None:
    """Register a runtime feature set for CSV-backed experiments."""
    CUSTOM_FEATURE_SETS[name] = list(feature_columns)
    CUSTOM_CATEGORICAL_COLUMNS[name] = list(categorical_columns or [])
