"""Load integrated Training/Validation CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ml.src import schema


def validate_columns(df: pd.DataFrame, feature_set: str = "core_11") -> None:
    """Validate required columns for a feature set."""
    required = set(schema.get_feature_columns(feature_set)) | {schema.TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _required_columns(feature_set: str) -> list[str]:
    return [*schema.get_feature_columns(feature_set), schema.TARGET_COLUMN]


def _read_family_csv(
    path: Path,
    source_family: str,
    nrows: int | None = None,
    feature_set: str = "core_11",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    df = pd.read_csv(path, nrows=nrows, usecols=_required_columns(feature_set), low_memory=False)
    df[schema.SOURCE_FAMILY_COLUMN] = source_family
    return df


def load_integrated_split(
    split: str,
    nrows_per_family: int | None = None,
    feature_set: str = "core_11",
) -> pd.DataFrame:
    """Load all source-family CSVs for a split into one dataframe."""
    if split == "train":
        files = schema.TRAIN_FILES
    elif split in {"validation", "valid", "val"}:
        files = schema.VALIDATION_FILES
    else:
        raise ValueError(f"Unsupported split: {split!r}")

    parts = [
        _read_family_csv(path, family, nrows_per_family, feature_set=feature_set)
        for family, path in files.items()
    ]
    df = pd.concat(parts, ignore_index=True)
    validate_columns(df, feature_set=feature_set)
    return df


def sample_integrated_split(
    split: str,
    sample_size: int | None,
    seed: int,
    feature_set: str = "core_11",
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Sample rows from full CSV files without loading all raw columns.

    The sample is balanced by source_family. Each family CSV is streamed in
    chunks and only the selected feature set plus target is read.
    """
    if sample_size is None:
        return load_integrated_split(split, feature_set=feature_set)

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if split == "train":
        files = schema.TRAIN_FILES
    elif split in {"validation", "valid", "val"}:
        files = schema.VALIDATION_FILES
    else:
        raise ValueError(f"Unsupported split: {split!r}")

    per_family = max(1, sample_size // len(files))
    remainder = sample_size % len(files)
    parts = []
    iterator = tqdm(
        list(files.items()),
        desc=f"Sampling {split}",
        unit="family",
        leave=False,
    )
    for idx, (family, path) in enumerate(iterator):
        family_size = per_family + (1 if idx < remainder else 0)
        parts.append(
            _sample_family_csv(
                path=path,
                source_family=family,
                sample_size=family_size,
                seed=seed + idx,
                feature_set=feature_set,
                chunksize=chunksize,
            )
        )
    df = pd.concat(parts, ignore_index=True)
    validate_columns(df, feature_set=feature_set)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def _sample_family_csv(
    path: Path,
    source_family: str,
    sample_size: int,
    seed: int,
    feature_set: str,
    chunksize: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    reservoir: pd.DataFrame | None = None
    rng_seed = seed
    chunk_iter = pd.read_csv(
        path,
        usecols=_required_columns(feature_set),
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        chunk_iter,
        desc=f"{source_family} reservoir",
        unit="chunk",
        leave=False,
    ):
        chunk = chunk.copy()
        chunk["_sample_key"] = pd.util.hash_pandas_object(
            chunk.index.to_series() + rng_seed,
            index=False,
        ).astype("uint64")
        candidates = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        reservoir = candidates.nsmallest(sample_size, "_sample_key").reset_index(drop=True)
        rng_seed += chunksize

    if reservoir is None:
        raise ValueError(f"CSV file is empty: {path}")

    reservoir = reservoir.drop(columns=["_sample_key"])
    reservoir[schema.SOURCE_FAMILY_COLUMN] = source_family
    return reservoir.reset_index(drop=True)


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Sample rows while preserving approximate source-family balance."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    frac = sample_size / len(df)
    sampled = (
        df.groupby(schema.SOURCE_FAMILY_COLUMN, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return sampled
