# Feature Combination Search Summary

Dataset: `openml_44964_superconductivity.csv`

Method:

- Candidate pool: top 60 features from LightGBM gain and permutation importance
- Search: importance-weighted random feature subsets
- Subset sizes: 10, 20, 30, 40
- Random candidates per size: 40
- Baseline model: LightGBM
- Split: random holdout, test_size=0.2, seed=42

## Best Combinations

| feature_count | best_experiment | RMSE | MAE | WAPE | R2 |
|---:|---|---:|---:|---:|---:|
| 10 | weighted_random_10_015 | 9.080337 | 5.412622 | 0.159322 | 0.928369 |
| 20 | weighted_random_20_034 | 8.821622 | 5.125779 | 0.150878 | 0.932393 |
| 30 | weighted_random_30_027 | 8.725013 | 5.053099 | 0.148739 | 0.933866 |
| 40 | weighted_random_40_036 | 8.689688 | 5.004281 | 0.147302 | 0.934400 |

Reference:

- All 81 features: RMSE 8.706177, MAE 4.970156, R2 0.934151
- Top 50 by gain: RMSE 8.734188, MAE 5.009522, R2 0.933727

## Interpretation

Feature combinations matter. The best 40-feature random combination beat the all-81 baseline on RMSE in this holdout split, while keeping about half the columns.

This is not proof of a globally optimal subset. It is a practical wrapper search result over a limited candidate pool. For a defensible final result, repeat across multiple random seeds or cross-validation folds.

## Recommendation

- Best compact subset: `config_best_40.json`
- Smaller but still strong subset: `config_best_30.json`
- Very compact subset: `config_best_20.json`

The 10-feature subset is useful for interpretation, but its RMSE gap is noticeably larger.
