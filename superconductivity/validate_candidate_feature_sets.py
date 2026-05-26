from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_selection_analysis import CSV_PATH, RESULT_DIR, TARGET, TEST_SIZE, make_model, wape


SEARCH_DIR = RESULT_DIR / "combination_search"
VALIDATION_DIR = RESULT_DIR / "candidate_validation"
SEEDS = [0, 1, 2, 3, 4]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(features: list[str], seed: int, X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    model = make_model()
    model.set_params(random_state=seed)
    model.fit(
        X_train[features],
        y_train,
        eval_set=[(X_valid[features], y_valid)],
        eval_metric="rmse",
        callbacks=[],
    )
    pred = model.predict(X_valid[features])
    return {
        "seed": seed,
        "feature_count": len(features),
        "rmse": rmse(y_valid, pred),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "wape": wape(y_valid, pred),
        "r2": float(r2_score(y_valid, pred)),
    }


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    all_features = [col for col in df.columns if col != TARGET]
    X = df[all_features]
    y = df[TARGET]

    selected = json.loads((SEARCH_DIR / "best_feature_sets.json").read_text(encoding="utf-8"))
    candidates: dict[str, list[str]] = {
        "all_81": all_features,
        "best_40": selected["best_40"]["features"],
        "best_30": selected["best_30"]["features"],
        "best_20": selected["best_20"]["features"],
        "best_10": selected["best_10"]["features"],
    }

    top50_config = json.loads((RESULT_DIR / "config_top_50_gain.json").read_text(encoding="utf-8"))
    corr_config = json.loads(
        (RESULT_DIR / "config_corr_filtered_0.95_gain_tiebreak.json").read_text(encoding="utf-8")
    )
    candidates["top_50_gain"] = top50_config["features"]
    candidates["corr_filtered_58"] = corr_config["features"]

    rows = []
    for name, features in candidates.items():
        for seed in SEEDS:
            row = evaluate(features, seed, X, y)
            row["candidate"] = name
            row["features"] = "|".join(features)
            rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(VALIDATION_DIR / "candidate_validation_by_seed.csv", index=False)

    summary = (
        detail.groupby("candidate")
        .agg(
            feature_count=("feature_count", "first"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            wape_mean=("wape", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
        .sort_values("rmse_mean")
    )
    summary.to_csv(VALIDATION_DIR / "candidate_validation_summary.csv", index=False)

    best_name = str(summary.iloc[0]["candidate"])
    best_features = candidates[best_name]
    best_config = {
        "csv": str(CSV_PATH),
        "target": TARGET,
        "model": "lightgbm",
        "features": best_features,
        "categorical": [],
        "test_size": TEST_SIZE,
        "seed": 42,
    }
    (VALIDATION_DIR / "recommended_config.json").write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (VALIDATION_DIR / "recommended_features.txt").write_text(
        "\n".join(best_features) + "\n",
        encoding="utf-8",
    )

    print("saved_dir=", VALIDATION_DIR)
    print(summary.to_string(index=False))
    print("recommended=", best_name)


if __name__ == "__main__":
    main()
