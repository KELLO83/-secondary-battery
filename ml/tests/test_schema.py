from __future__ import annotations

from ml.src import schema


def test_nasa_default_feature_set() -> None:
    assert schema.DEFAULT_FEATURE_SET == "discharge_summary"
    assert schema.get_feature_columns() == schema.NASA_FEATURE_SETS["discharge_summary"]
    assert schema.TARGET_COLUMN == "capacity"


def test_nasa_feature_sets_are_ordered_by_information() -> None:
    cycle_basic = set(schema.get_feature_columns("cycle_basic"))
    discharge_summary = set(schema.get_feature_columns("discharge_summary"))
    discharge_health = set(schema.get_feature_columns("discharge_health"))

    assert cycle_basic < discharge_summary
    assert discharge_summary < discharge_health
    assert "soh" not in discharge_summary
    assert "soh" in discharge_health


def test_battery_id_is_group_key_and_categorical_feature() -> None:
    assert schema.SOURCE_FAMILY_COLUMN == "battery_id"
    assert "battery_id" in schema.get_categorical_columns("discharge_summary")
