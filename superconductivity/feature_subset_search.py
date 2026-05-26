from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_selection_analysis import CSV_PATH, RESULT_DIR, SEED, TARGET, TEST_SIZE, make_model, wape


SEARCH_DIR = RESULT_DIR / "combination_search"
SUBSET_SIZES = [10, 20, 30, 40]
N_RANDOM_CANDIDATES = 40
POOL_SIZE = 60


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(features: list[str], X_train, X_valid, y_train, y_valid) -> dict:
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
        "feature_count": len(features),
        "rmse": rmse(y_valid, pred),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "wape": wape(y_valid, pred),
        "r2": float(r2_score(y_valid, pred)),
        "features": "|".join(features),
    }


def main() -> None:
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    all_features = [col for col in df.columns if col != TARGET]
    X = df[all_features]
    y = df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    gain = pd.read_csv(RESULT_DIR / "lightgbm_gain_split_importance.csv")
    perm = pd.read_csv(RESULT_DIR / "permutation_importance_rmse.csv")
    gain_order = gain["feature"].tolist()
    perm_order = perm["feature"].tolist()

    candidate_pool = list(dict.fromkeys(gain_order[:POOL_SIZE] + perm_order[:POOL_SIZE]))
    score = {feature: 0.0 for feature in candidate_pool}
    for rank, feature in enumerate(gain_order):
        if feature in score:
            score[feature] += 1.0 / (rank + 1)
    for rank, feature in enumerate(perm_order):
        if feature in score:
            score[feature] += 1.0 / (rank + 1)

    weights = np.array([score[feature] for feature in candidate_pool], dtype=float)
    weights = weights / weights.sum()

    rng = np.random.default_rng(SEED)
    results: list[dict] = []
    best_sets: dict[str, dict] = {}

    for subset_size in SUBSET_SIZES:
        candidates: dict[str, list[str]] = {
            f"top_{subset_size}_gain": gain_order[:subset_size],
            f"top_{subset_size}_permutation": perm_order[:subset_size],
        }

        for idx in range(N_RANDOM_CANDIDATES):
            sampled = rng.choice(
                candidate_pool,
                size=subset_size,
                replace=False,
                p=weights,
            ).tolist()
            # Stable ordering keeps generated configs readable.
            sampled = sorted(sampled, key=lambda feature: gain_order.index(feature))
            candidates[f"weighted_random_{subset_size}_{idx + 1:03d}"] = sampled

        for name, features in candidates.items():
            row = evaluate(features, X_train, X_valid, y_train, y_valid)
            row["experiment"] = name
            results.append(row)

        size_df = pd.DataFrame([row for row in results if row["feature_count"] == subset_size])
        best = size_df.sort_values("rmse").iloc[0].to_dict()
        best_sets[f"best_{subset_size}"] = {
            "experiment": best["experiment"],
            "feature_count": int(best["feature_count"]),
            "rmse": float(best["rmse"]),
            "mae": float(best["mae"]),
            "wape": float(best["wape"]),
            "r2": float(best["r2"]),
            "features": best["features"].split("|"),
        }

    result_df = pd.DataFrame(results).sort_values(["feature_count", "rmse"])
    result_df.to_csv(SEARCH_DIR / "combination_search_metrics.csv", index=False)

    (SEARCH_DIR / "best_feature_sets.json").write_text(
        json.dumps(best_sets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for key, value in best_sets.items():
        config = {
            "csv": str(CSV_PATH),
            "target": TARGET,
            "model": "lightgbm",
            "features": value["features"],
            "categorical": [],
            "test_size": TEST_SIZE,
            "seed": SEED,
        }
        (SEARCH_DIR / f"config_{key}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("saved_dir=", SEARCH_DIR)
    print(result_df.groupby("feature_count").head(5)[["feature_count", "experiment", "rmse", "mae", "wape", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
