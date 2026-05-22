from __future__ import annotations

from ml.src import schema


def test_source_family_is_metadata_not_feature() -> None:
    assert schema.SOURCE_FAMILY_COLUMN in schema.METADATA_COLUMNS
    assert schema.SOURCE_FAMILY_COLUMN not in schema.OFFICIAL_FEATURE_COLUMNS
    assert schema.SOURCE_FAMILY_COLUMN not in schema.CORE_11_FEATURE_COLUMNS


def test_core_11_is_default_feature_set() -> None:
    assert schema.get_feature_columns() == schema.CORE_11_FEATURE_COLUMNS
    assert len(schema.CORE_11_FEATURE_COLUMNS) == 11


def test_expanded_feature_sets_add_expected_columns() -> None:
    design_added = set(schema.DESIGN_15_FEATURE_COLUMNS) - set(schema.CORE_11_FEATURE_COLUMNS)
    chem_added = set(schema.CHEM_22_FEATURE_COLUMNS) - set(schema.DESIGN_15_FEATURE_COLUMNS)

    assert design_added == {
        "sintering_T1(C)",
        "sintering_t1(h)",
        "measurement_T(C)",
        "C-rate",
    }
    assert chem_added == {
        "Li_fraction",
        "Ni_fraction",
        "Mn_fraction",
        "Co_fraction",
        "dopant_fraction",
        "active_proportion",
        "binder_proportion",
    }
