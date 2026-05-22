from __future__ import annotations

from ml.src import schema
from ml.src.data.loader import sample_integrated_split


def test_load_nasa_train_validation_group_split() -> None:
    train = sample_integrated_split("train", sample_size=None, seed=42, feature_set="discharge_summary")
    valid = sample_integrated_split("validation", sample_size=None, seed=42, feature_set="discharge_summary")

    assert len(train) > 0
    assert len(valid) > 0
    assert schema.TARGET_COLUMN in train.columns
    assert set(train[schema.SOURCE_FAMILY_COLUMN]).isdisjoint(set(valid[schema.SOURCE_FAMILY_COLUMN]))


def test_nasa_sample_split_keeps_requested_size() -> None:
    train = sample_integrated_split("train", sample_size=50, seed=42, feature_set="cycle_basic")
    assert len(train) == 50
    assert set(schema.get_feature_columns("cycle_basic")).issubset(train.columns)
