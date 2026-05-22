"""Load NASA cycle-level battery data."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.src import schema
from ml.src.data.nasa_cycle_builder import build_nasa_cycle_level_dataset


def validate_columns(df: pd.DataFrame, feature_set: str = schema.DEFAULT_FEATURE_SET) -> None:
    """Validate required columns for a feature set."""
    required = set(schema.get_feature_columns(feature_set)) | {schema.TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_nasa_cycle_level(force_rebuild: bool = False) -> pd.DataFrame:
    """Load or build the processed NASA discharge cycle-level table."""
    df = build_nasa_cycle_level_dataset(force=force_rebuild)
    df[schema.TARGET_COLUMN] = pd.to_numeric(df[schema.TARGET_COLUMN], errors="coerce")
    return df.dropna(subset=[schema.TARGET_COLUMN, schema.SOURCE_FAMILY_COLUMN]).reset_index(drop=True)


def load_integrated_split(
    split: str,
    nrows_per_family: int | None = None,
    feature_set: str = schema.DEFAULT_FEATURE_SET,
) -> pd.DataFrame:
    """Compatibility wrapper for old callers.

    NASA does not ship separate train/validation files. Use
    ``sample_integrated_split`` so the split is created by battery_id groups.
    """
    del nrows_per_family
    return sample_integrated_split(split=split, sample_size=None, seed=42, feature_set=feature_set)


def sample_integrated_split(
    split: str,
    sample_size: int | None,
    seed: int,
    feature_set: str = schema.DEFAULT_FEATURE_SET,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Return a train or validation split sampled from NASA cycle-level data.

    Splitting is performed by ``battery_id`` so validation batteries are unseen
    cells, preventing cycle-level leakage from the same battery appearing in
    both train and validation.
    """
    del chunksize
    df = load_nasa_cycle_level()
    validate_columns(df, feature_set=feature_set)
    train_idx, valid_idx = _battery_group_split(df, seed=seed)
    if split == "train":
        part = df.iloc[train_idx].copy()
    elif split in {"validation", "valid", "val"}:
        part = df.iloc[valid_idx].copy()
    else:
        raise ValueError(f"Unsupported split: {split!r}")

    if sample_size is not None:
        part = sample_frame(part, sample_size=sample_size, seed=seed)
    return part.reset_index(drop=True)


def _battery_group_split(df: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(df, groups=df[schema.SOURCE_FAMILY_COLUMN]))
    return pd.Index(train_idx), pd.Index(valid_idx)


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Sample rows while preserving approximate battery_id balance."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    return (
        df.groupby(schema.SOURCE_FAMILY_COLUMN, group_keys=False)
        .sample(frac=sample_size / len(df), random_state=seed)
        .sample(n=min(sample_size, len(df)), random_state=seed)
        .reset_index(drop=True)
    )
