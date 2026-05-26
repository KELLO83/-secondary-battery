# Generic Tabular Regression Data Contract

## Input

The training input is a CSV file.

Required:

- `--csv`: path to CSV
- `--target`: numeric regression target column

Optional:

- `--features`: exact comma-separated feature list
- `--exclude`: columns to remove from automatic feature selection
- `--categorical`: columns to treat as categorical
- `--config`: JSON config containing the same settings

## Feature Selection

If `features` is provided, only those columns are used.

If `features` is not provided:

```text
features = all columns - target - exclude - unnamed index columns
```

Target-derived columns must be excluded manually. Examples:

- target itself
- normalized target
- residual target
- post-measurement label
- future value

## Config Format

```json
{
  "csv": "path/to/data.csv",
  "target": "target_column",
  "model": "lightgbm",
  "features": ["f1", "f2", "f3"],
  "exclude": ["id", "leakage_column"],
  "categorical": ["category_column"],
  "test_size": 0.2,
  "seed": 42,
  "model_params": {
    "n_estimators": 100
  }
}
```

## Output

Generic runs write:

```text
results/tabular_regression_experiments.csv
```

Optional predictions:

```powershell
--save-predictions
```
