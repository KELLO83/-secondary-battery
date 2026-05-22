from __future__ import annotations

from ml.src import schema
from ml.src.data.loader import load_integrated_split


def test_load_small_integrated_train_split() -> None:
    df = load_integrated_split("train", nrows_per_family=2, feature_set="full")
    assert len(df) == 8
    assert schema.SOURCE_FAMILY_COLUMN in df.columns
    assert set(df[schema.SOURCE_FAMILY_COLUMN]) == set(schema.EXPECTED_SOURCE_FAMILIES)
