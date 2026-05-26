# Candidate Feature Set Validation

Dataset: `openml_44964_superconductivity.csv`

Validation:

- Candidates: all 81, top 50 gain, correlation-filtered 58, searched best 40/30/20/10
- Seeds: 0, 1, 2, 3, 4
- Split: random holdout, test_size=0.2
- Model: LightGBM

## Result

| candidate | features | RMSE mean | RMSE std | MAE mean | WAPE mean | R2 mean |
|---|---:|---:|---:|---:|---:|---:|
| all_81 | 81 | 9.227177 | 0.274352 | 5.198675 | 0.150110 | 0.927288 |
| top_50_gain | 50 | 9.254649 | 0.311122 | 5.236718 | 0.151209 | 0.926837 |
| corr_filtered_58 | 58 | 9.260771 | 0.336509 | 5.234264 | 0.151139 | 0.926725 |
| best_30 | 30 | 9.278601 | 0.291944 | 5.294876 | 0.152902 | 0.926464 |
| best_40 | 40 | 9.278733 | 0.321207 | 5.265789 | 0.152051 | 0.926452 |
| best_20 | 20 | 9.379023 | 0.373139 | 5.358808 | 0.154730 | 0.924826 |
| best_10 | 10 | 9.666728 | 0.377073 | 5.640374 | 0.162847 | 0.920165 |

## Recommendation

Use all 81 features for final model-comparison experiments.

Reason:

- It has the best average RMSE across five seeds.
- Feature-reduced candidates are close, but they do not consistently beat the full feature set.
- The previously strong `best_40` result was likely partly split-specific.

For a compact secondary experiment, use `top_50_gain`. It is slightly weaker than all 81 features but still close and easier to interpret.

Saved recommendation:

- `recommended_config.json`: all 81 features
- `recommended_features.txt`: selected feature names
