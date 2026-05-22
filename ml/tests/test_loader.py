from __future__ import annotations

from ml.src import schema
from ml.src.data.loader import _allocate_family_sample_sizes, load_integrated_split


def test_load_small_integrated_train_split() -> None:
    df = load_integrated_split("train", nrows_per_family=2, feature_set="core_11")
    assert len(df) == 8
    assert schema.SOURCE_FAMILY_COLUMN in df.columns
    assert set(df[schema.SOURCE_FAMILY_COLUMN]) == set(schema.EXPECTED_SOURCE_FAMILIES)
    assert not set(schema.TARGET_DERIVED_LEAKAGE_COLUMNS) & set(df.columns)


def test_load_small_chem_derived_train_split() -> None:
    df = load_integrated_split("train", nrows_per_family=2, feature_set="chem_derived")
    assert len(df) == 8
    assert "voltage_window" in df.columns
    assert "Ni_to_Mn" in df.columns
    assert "active_to_binder" in df.columns
    assert not set(schema.TARGET_DERIVED_LEAKAGE_COLUMNS) & set(df.columns)


def test_sample_allocation_preserves_source_family_ratio() -> None:
    sizes = _allocate_family_sample_sizes(1_000, schema.TRAIN_ROW_COUNTS)
    assert sum(sizes.values()) == 1_000
    assert sizes["NCM"] > sizes["Others"] > sizes["NCA"] > sizes["LFP"]
