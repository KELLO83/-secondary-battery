# AGENT: Generic Tabular Regression Experiment Rules

## 1. Scope

This project is for comparing models on tabular regression datasets.

In scope:

- CSV-based tabular regression experiments
- target/feature selection
- preprocessing for numeric and categorical columns
- single-model training scripts
- regression metrics and experiment logs
- model comparison documentation

Out of scope unless explicitly requested:

- frontend
- backend/API server
- database service
- production deployment
- AutoML sweeps or multi-model batch runners

## 2. Primary Entry Point

Use root `train.py` for one experiment.

Generic CSV mode:

```powershell
.\.venv314\Scripts\python.exe train.py `
  --csv "path\to\data.csv" `
  --target target_column `
  --exclude id,leakage_column `
  --model lightgbm
```

Rules:

- `--csv` and `--target` switch `train.py` into generic tabular regression mode.
- Without `--csv`, legacy dataset-specific modes may still exist but are not the project default.
- One command must run exactly one model on one dataset/target/split/seed.

## 3. Dataset Policy

The active dataset is not fixed. Any dataset can be used if it satisfies:

- tabular structure readable as CSV
- one numeric regression target column
- feature columns available before prediction time
- leakage columns explicitly excluded

For the current AI Hub material-property sample:

```text
Experimental materials property data/02.라벨링데이터/Resistivity_data.csv
Experimental materials property data/02.라벨링데이터/Hardness_data.csv
```

Recommended tasks:

- `Resistivity` regression
- `Hardness` regression

Exclude IDs, coordinates, strings, and target-derived columns when they are not valid prediction-time features.

## 4. Runtime Environments

Use `.venv314` for the general model comparison path:

- LightGBM
- CatBoost
- TabPFN v3
- neural / transformer / foundation models

Use `.venv314t` only for CPU-focused LightGBM experiments when package compatibility is confirmed.

Do not source-build PyTorch, LightGBM, CatBoost, CUDA packages, or TabPFN dependencies unless explicitly requested.

## 5. Model Rules

Supported generic CSV models in `train.py`:

- `dummy_mean`
- `dummy_median`
- `ridge`
- `lightgbm`
- `catboost`
- `realmlp`
- `tabm`
- `tabr`
- `dcnv2`
- `node`
- `ft_transformer`
- `tab_transformer`
- `tabnet`
- `tabpfn`
- `tabiclv2`

Some models require optional packages, GPU support, tokens, or checkpoints. Config compatibility does not guarantee the runtime environment has every dependency installed.

TabPFN v3:

- pretrained/in-context model
- do not train from scratch
- use `TABPFN_TOKEN`, `.secrets/tabpfn_token`, cached token files, or explicit package-supported checkpoint access
- do not rely on interactive login for unattended runs

## 6. Evaluation

Default split:

- random holdout
- `--test-size 0.2`
- `--seed 42`

Use group/time split only when the dataset requires it. If a dataset has repeated entities, time ordering, or leakage-prone groups, document the split key before running.

Default metrics:

- RMSE
- MAE
- MAPE
- WAPE

MAPE is diagnostic only when targets can be zero or near zero.

## 7. Logging And Results

Generic CSV mode writes to:

```text
results/tabular_regression_experiments.csv
```

Rules:

- do not run multiple commands that append to the same CSV in parallel
- log dataset path, target, excluded columns, feature count, split size, model params, metrics
- use `--save-predictions` only when prediction inspection is needed

## 8. Experiment Unit

One run means:

```text
1 dataset + 1 target + 1 model + 1 split seed + 1 parameter setting
```

Do not add automatic sweeps, model loops, leaderboard runners, or multi-seed loops to `train.py`.
