"""Random and group split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_random_split(
    df: pd.DataFrame,
    seed: int,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows randomly into train/valid/test."""
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1.0")
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=seed)
    rel_valid = valid_size / (valid_size + test_size)
    valid_df, test_df = train_test_split(temp_df, train_size=rel_valid, random_state=seed)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_group_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by non-overlapping groups."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(first.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    rel_valid = valid_size / (valid_size + test_size)
    second = GroupShuffleSplit(n_splits=1, train_size=rel_valid, random_state=seed)
    valid_idx, test_idx = next(second.split(temp_df, groups=temp_df[group_col]))
    valid_df = temp_df.iloc[valid_idx]
    test_df = temp_df.iloc[test_idx]

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)
