# System Architecture

## Main Flow

```text
CSV file
  -> train.py --csv/--target or train.py --config
  -> generic tabular loader
  -> feature/target split
  -> sklearn preprocessing
  -> model fit/predict
  -> regression metrics
  -> results/tabular_regression_experiments.csv
```

## Key Files

```text
train.py
ml/scripts/run_tabular_regression.py
configs/*.json
results/tabular_regression_experiments.csv
```

## Compatibility

Legacy dataset-specific code can remain for reference, but new experiments should use the generic CSV interface unless the user explicitly asks for a dataset-specific pipeline.
