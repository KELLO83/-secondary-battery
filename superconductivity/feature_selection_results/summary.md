# Superconductivity Feature Selection Summary

Dataset: `openml_44964_superconductivity.csv`

- Rows: 21,263
- Columns: 82
- Target: `critical_temp`
- Input features: 81
- Split: random holdout, test_size=0.2, seed=42
- Baseline model: LightGBM regression

## Main Result

The best validation score came from using all 81 features.

| experiment | features | RMSE | MAE | WAPE | R2 |
|---|---:|---:|---:|---:|---:|
| all_81_features | 81 | 8.706177 | 4.970156 | 0.146297 | 0.934151 |
| top_50_gain | 50 | 8.734188 | 5.009522 | 0.147456 | 0.933727 |
| corr_filtered_0.95_gain_tiebreak | 58 | 8.755005 | 5.005775 | 0.147346 | 0.933410 |
| top_50_permutation | 50 | 8.761016 | 5.022120 | 0.147827 | 0.933319 |
| top_30_permutation | 30 | 8.821841 | 5.104235 | 0.150244 | 0.932390 |

Interpretation: feature reduction did not improve validation RMSE in this split. However, top 50 features by LightGBM gain are very close to the full-feature baseline and are a reasonable compact feature set.

## Most Important Features

Top features by LightGBM gain:

1. `range_ThermalConductivity`
2. `wtd_gmean_ThermalConductivity`
3. `range_atomic_radius`
4. `std_atomic_mass`
5. `wtd_mean_ThermalConductivity`
6. `wtd_std_ElectronAffinity`
7. `wtd_gmean_Valence`
8. `wtd_mean_Valence`
9. `gmean_ElectronAffinity`
10. `std_Density`

Top features by permutation importance:

1. `range_ThermalConductivity`
2. `range_atomic_radius`
3. `wtd_gmean_ThermalConductivity`
4. `std_atomic_mass`
5. `wtd_gmean_Valence`
6. `wtd_std_ElectronAffinity`
7. `wtd_mean_Valence`
8. `wtd_std_Valence`
9. `gmean_ElectronAffinity`
10. `wtd_mean_ThermalConductivity`

The two methods agree strongly on the most important signals: thermal conductivity spread, atomic radius range, weighted thermal conductivity, atomic mass variation, valence, and electron affinity.

## Group-Only Result

Best single feature group:

| experiment | features | RMSE | R2 |
|---|---:|---:|---:|
| group_Density | 10 | 9.830383 | 0.916047 |
| group_atomic_radius | 10 | 10.002719 | 0.913078 |
| group_atomic_mass | 10 | 10.096192 | 0.911446 |
| group_ElectronAffinity | 10 | 10.166369 | 0.910210 |
| group_fie | 10 | 10.264779 | 0.908464 |

Single groups are weaker than using the full feature set, so the target depends on combined material-property statistics.

## Saved Files

- `feature_subset_metrics.csv`: all subset retraining metrics
- `lightgbm_gain_split_importance.csv`: LightGBM gain/split importance
- `permutation_importance_rmse.csv`: permutation importance by RMSE increase
- `selected_feature_sets.json`: selected feature lists
- `config_*.json`: ready-to-run LightGBM configs for each feature subset
- `feature_selection_analysis.py`: reproducible analysis script in the parent folder

## Recommendation

For best performance, use all 81 features.

For a compact model, use `config_top_50_gain.json`. It uses 50 features and only loses about 0.028 RMSE versus the full 81-feature baseline on this split.
