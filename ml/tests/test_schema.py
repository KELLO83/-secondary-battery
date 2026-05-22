from __future__ import annotations

from ml.src import schema


def test_source_family_is_metadata_not_feature() -> None:
    assert schema.SOURCE_FAMILY_COLUMN in schema.METADATA_COLUMNS
    assert schema.SOURCE_FAMILY_COLUMN not in schema.CORE_11_FEATURE_COLUMNS
    assert schema.SOURCE_FAMILY_COLUMN not in schema.CHEM_22_FEATURE_COLUMNS


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


def test_target_derived_leakage_columns_are_not_supported_features() -> None:
    supported_columns = set()
    for feature_set in schema.SUPPORTED_FEATURE_SETS:
        supported_columns.update(schema.get_feature_columns(feature_set))

    assert not set(schema.TARGET_DERIVED_LEAKAGE_COLUMNS) & supported_columns


def test_chem_derived_extends_chem_22_without_leakage() -> None:
    chem_22 = set(schema.get_feature_columns("chem_22"))
    chem_derived = set(schema.get_feature_columns("chem_derived"))
    assert chem_22 < chem_derived
    assert {
        "voltage_window",
        "voltage_mid",
        "Ni_to_Mn",
        "Ni_to_Co",
        "Li_to_TM",
        "active_to_binder",
        "total_transition_metal",
    }.issubset(chem_derived)


def test_official_feature_set_is_not_supported() -> None:
    try:
        schema.get_feature_columns("official")
    except ValueError as exc:
        assert "Unsupported feature_set" in str(exc)
    else:
        raise AssertionError("official feature set must not be supported")
