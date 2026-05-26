# Evaluation Protocol

## Default Split

Generic CSV mode uses random holdout:

```text
test_size = 0.2
seed = 42
```

Use a different split only when the data requires it.

Examples:

- entity/group holdout for repeated subjects, users, machines, cells, regions
- time-based split for forecasting
- site/batch holdout for domain generalization

## Metrics

Default metrics:

- RMSE
- MAE
- MAPE
- WAPE

Primary ranking should usually use RMSE or MAE. MAPE is unstable when targets are zero or near zero.

## Leakage Rules

Do not include:

- target column
- target-derived columns
- IDs that encode row order or label generation
- future information
- post-outcome measurements unavailable at prediction time

All excluded columns should be visible in result logs.
