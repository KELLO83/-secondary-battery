from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "openml_44964_superconductivity.csv"
RESULT_DIR = ROOT / "feature_selection_results"
TARGET = "critical_temp"
SEED = 42
TEST_SIZE = 0.2
TOP_NS = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def make_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def evaluate(name: str, features: list[str], X_train, X_valid, y_train, y_valid) -> dict:
    model = make_model()
    model.fit(
        X_train[features],
        y_train,
        eval_set=[(X_valid[features], y_valid)],
        eval_metric="rmse",
        callbacks=[],
    )
    pred = model.predict(X_valid[features])
    return {
        "experiment": name,
        "feature_count": len(features),
        "rmse": rmse(y_valid, pred),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "wape": wape(y_valid, pred),
        "r2": float(r2_score(y_valid, pred)),
        "features": "|".join(features),
    }


def feature_group(feature: str) -> str:
    for group in [
        "atomic_mass",
        "fie",
        "atomic_radius",
        "Density",
        "ElectronAffinity",
        "FusionHeat",
        "ThermalConductivity",
        "Valence",
    ]:
        if group in feature:
            return group
    if feature == "number_of_elements":
        return "number_of_elements"
    return "other"


def correlation_filtered_features(
    X_train: pd.DataFrame,
    features: list[str],
    gain_rank: dict[str, int],
    threshold: float = 0.95,
) -> list[str]:
    corr = X_train[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()
    for col in upper.columns:
        correlated = [idx for idx, value in upper[col].dropna().items() if value > threshold]
        for row in correlated:
            keep, drop = (row, col) if gain_rank[row] < gain_rank[col] else (col, row)
            to_drop.add(drop)
    return [feature for feature in features if feature not in to_drop]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    features = [col for col in df.columns if col != TARGET]
    X = df[features]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    metadata = {
        "csv": str(CSV_PATH),
        "target": TARGET,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(features),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "train_rows": int(X_train.shape[0]),
        "valid_rows": int(X_valid.shape[0]),
        "all_numeric": bool(all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)),
        "missing_total": int(df.isna().sum().sum()),
    }
    (RESULT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline_model = make_model()
    baseline_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[],
    )
    baseline_pred = baseline_model.predict(X_valid)
    baseline_metrics = {
        "experiment": "all_81_features",
        "feature_count": len(features),
        "rmse": rmse(y_valid, baseline_pred),
        "mae": float(mean_absolute_error(y_valid, baseline_pred)),
        "wape": wape(y_valid, baseline_pred),
        "r2": float(r2_score(y_valid, baseline_pred)),
        "features": "|".join(features),
    }

    booster = baseline_model.booster_
    importance = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }
    )
    importance["gain_rank"] = importance["gain_importance"].rank(
        method="first", ascending=False
    ).astype(int)
    importance["split_rank"] = importance["split_importance"].rank(
        method="first", ascending=False
    ).astype(int)
    importance = importance.sort_values("gain_importance", ascending=False)
    importance.to_csv(RESULT_DIR / "lightgbm_gain_split_importance.csv", index=False)

    perm = permutation_importance(
        baseline_model,
        X_valid,
        y_valid,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    permutation = pd.DataFrame(
        {
            "feature": features,
            "rmse_increase_mean": perm.importances_mean,
            "rmse_increase_std": perm.importances_std,
        }
    )
    permutation["permutation_rank"] = permutation["rmse_increase_mean"].rank(
        method="first", ascending=False
    ).astype(int)
    permutation = permutation.sort_values("rmse_increase_mean", ascending=False)
    permutation.to_csv(RESULT_DIR / "permutation_importance_rmse.csv", index=False)

    gain_features = importance["feature"].tolist()
    permutation_features = permutation["feature"].tolist()
    gain_rank = {feature: rank for rank, feature in enumerate(gain_features)}

    experiments = [baseline_metrics]
    selected_sets: dict[str, list[str]] = {"all_81_features": features}

    for n in TOP_NS:
        selected_sets[f"top_{n}_gain"] = gain_features[:n]
        selected_sets[f"top_{n}_permutation"] = permutation_features[:n]

    corr_features = correlation_filtered_features(X_train, features, gain_rank, threshold=0.95)
    selected_sets["corr_filtered_0.95_gain_tiebreak"] = corr_features

    for group in sorted({feature_group(feature) for feature in features}):
        group_features = [feature for feature in features if feature_group(feature) == group]
        selected_sets[f"group_{group}"] = group_features

    for name, selected_features in selected_sets.items():
        if name == "all_81_features":
            continue
        experiments.append(
            evaluate(name, selected_features, X_train, X_valid, y_train, y_valid)
        )

    metrics = pd.DataFrame(experiments).sort_values("rmse")
    metrics.to_csv(RESULT_DIR / "feature_subset_metrics.csv", index=False)

    selected_json = {
        name: {
            "feature_count": len(selected_features),
            "features": selected_features,
        }
        for name, selected_features in selected_sets.items()
    }
    (RESULT_DIR / "selected_feature_sets.json").write_text(
        json.dumps(selected_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name, selected_features in selected_sets.items():
        config = {
            "csv": str(CSV_PATH),
            "target": TARGET,
            "model": "lightgbm",
            "features": selected_features,
            "categorical": [],
            "test_size": TEST_SIZE,
            "seed": SEED,
        }
        config_path = RESULT_DIR / f"config_{name}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved_dir=", RESULT_DIR)
    print("best_metrics")
    print(metrics[["experiment", "feature_count", "rmse", "mae", "wape", "r2"]].head(12).to_string(index=False))
    print("top_gain")
    print(importance[["feature", "gain_importance", "split_importance"]].head(20).to_string(index=False))
    print("top_permutation")
    print(permutation[["feature", "rmse_increase_mean", "rmse_increase_std"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
