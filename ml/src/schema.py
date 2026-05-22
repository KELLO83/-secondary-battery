"""Dataset schema constants for AI Hub battery data."""

from __future__ import annotations

from pathlib import Path

TARGET_COLUMN = "remain_capacity"
SOURCE_FAMILY_COLUMN = "source_family"

TRAIN_FILES = {
    "LFP": Path("Training/LFP_train_dataset.csv"),
    "NCA": Path("Training/NCA_train_dataset.csv"),
    "NCM": Path("Training/NCM_train_dataset.csv"),
    "Others": Path("Training/others_train_dataset.csv"),
}

TRAIN_ROW_COUNTS = {
    "LFP": 131_790,
    "NCA": 769_225,
    "NCM": 8_750_613,
    "Others": 3_409_068,
}

VALIDATION_FILES = {
    "LFP": Path("Validation/LFP_val_dataset.csv"),
    "NCA": Path("Validation/NCA_val_dataset.csv"),
    "NCM": Path("Validation/NCM_val_dataset.csv"),
    "Others": Path("Validation/others_val_dataset.csv"),
}

VALIDATION_ROW_COUNTS = {
    "LFP": 16_474,
    "NCA": 96_153,
    "NCM": 1_093_827,
    "Others": 426_133,
}

CORE_NUMERIC_COLUMNS = [
    "voltage_range(V)_min",
    "voltage_range(V)_max",
]

CORE_CATEGORICAL_COLUMNS = [
    "material_structure",
    "synthesis_method",
    "Li_source",
    "Ni_source",
    "Co_source",
    "Mn_source",
    "electrolyte",
    "separator",
    "counter_electrode",
]

EXTENDED_NUMERIC_COLUMNS = [
    "sintering_T1(C)",
    "sintering_t1(h)",
    "measurement_T(C)",
    "Li_fraction",
    "Ni_fraction",
    "Mn_fraction",
    "Co_fraction",
    "dopant_fraction",
    "active_proportion",
    "binder_proportion",
    "particle_size(um)",
    "C-rate",
    "discharge_capacity (mAh/g)",
    "Strain",
    "state_of_charge",
    "length_a",
    "length_b",
    "length_c",
    "angle_alpha",
    "angle_beta",
    "angle_gamma",
    "volume",
    "density",
    "interlayer_dist",
    "energy",
    "tm_o_bond_length",
    "perc_barrier_1d",
    "perc_barrier_2d",
    "perc_radius_1d",
    "perc_radius_2d",
    "max_packing_eff",
    "chemical_ordering",
    "struct_hetero_bond",
    "struct_hetero_cell",
]

TARGET_DERIVED_LEAKAGE_COLUMNS = [
    "discharge_capacity (mAh/g)",
    "state_of_charge",
]

EXTENDED_CATEGORICAL_COLUMNS = [
    "space_group_symbol",
]

METADATA_COLUMNS = [
    "material_id",
    "chemical_formula",
    "DOI",
    "journal_name",
    "Class",
    "Unnamed: 0",
    SOURCE_FAMILY_COLUMN,
]

CORE_FEATURE_COLUMNS = CORE_NUMERIC_COLUMNS + CORE_CATEGORICAL_COLUMNS
CORE_11_NUMERIC_COLUMNS = CORE_NUMERIC_COLUMNS
CORE_11_CATEGORICAL_COLUMNS = CORE_CATEGORICAL_COLUMNS
CORE_11_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS
DESIGN_15_NUMERIC_COLUMNS = CORE_NUMERIC_COLUMNS + [
    "sintering_T1(C)",
    "sintering_t1(h)",
    "measurement_T(C)",
    "C-rate",
]
DESIGN_15_CATEGORICAL_COLUMNS = CORE_CATEGORICAL_COLUMNS
DESIGN_15_FEATURE_COLUMNS = DESIGN_15_NUMERIC_COLUMNS + DESIGN_15_CATEGORICAL_COLUMNS
CHEM_22_NUMERIC_COLUMNS = DESIGN_15_NUMERIC_COLUMNS + [
    "Li_fraction",
    "Ni_fraction",
    "Mn_fraction",
    "Co_fraction",
    "dopant_fraction",
    "active_proportion",
    "binder_proportion",
]
CHEM_22_CATEGORICAL_COLUMNS = CORE_CATEGORICAL_COLUMNS
CHEM_22_FEATURE_COLUMNS = CHEM_22_NUMERIC_COLUMNS + CHEM_22_CATEGORICAL_COLUMNS
CHEM_DERIVED_NUMERIC_COLUMNS = CHEM_22_NUMERIC_COLUMNS + [
    "voltage_window",
    "voltage_mid",
    "Ni_to_Mn",
    "Ni_to_Co",
    "Li_to_TM",
    "active_to_binder",
    "total_transition_metal",
]
CHEM_DERIVED_CATEGORICAL_COLUMNS = CORE_CATEGORICAL_COLUMNS
CHEM_DERIVED_FEATURE_COLUMNS = CHEM_DERIVED_NUMERIC_COLUMNS + CHEM_DERIVED_CATEGORICAL_COLUMNS

REQUIRED_COLUMNS = sorted(set([TARGET_COLUMN, *CORE_FEATURE_COLUMNS]))
EXPECTED_SOURCE_FAMILIES = ("LFP", "NCA", "NCM", "Others")
SUPPORTED_FEATURE_SETS = ("core_11", "design_15", "chem_22", "chem_derived")


def get_feature_columns(feature_set: str = "core_11") -> list[str]:
    """Return feature columns for a named feature set."""
    if feature_set in {"core", "core_11"}:
        return list(CORE_11_FEATURE_COLUMNS)
    if feature_set == "design_15":
        return list(DESIGN_15_FEATURE_COLUMNS)
    if feature_set == "chem_22":
        return list(CHEM_22_FEATURE_COLUMNS)
    if feature_set == "chem_derived":
        return list(CHEM_DERIVED_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported feature_set: {feature_set!r}")


def get_numeric_columns(feature_set: str = "core_11") -> list[str]:
    """Return numeric feature columns for a named feature set."""
    if feature_set in {"core", "core_11"}:
        return list(CORE_11_NUMERIC_COLUMNS)
    if feature_set == "design_15":
        return list(DESIGN_15_NUMERIC_COLUMNS)
    if feature_set == "chem_22":
        return list(CHEM_22_NUMERIC_COLUMNS)
    if feature_set == "chem_derived":
        return list(CHEM_DERIVED_NUMERIC_COLUMNS)
    raise ValueError(f"Unsupported feature_set: {feature_set!r}")


def get_categorical_columns(feature_set: str = "core_11") -> list[str]:
    """Return categorical feature columns for a named feature set."""
    if feature_set in {"core", "core_11"}:
        return list(CORE_11_CATEGORICAL_COLUMNS)
    if feature_set == "design_15":
        return list(DESIGN_15_CATEGORICAL_COLUMNS)
    if feature_set == "chem_22":
        return list(CHEM_22_CATEGORICAL_COLUMNS)
    if feature_set == "chem_derived":
        return list(CHEM_DERIVED_CATEGORICAL_COLUMNS)
    raise ValueError(f"Unsupported feature_set: {feature_set!r}")
