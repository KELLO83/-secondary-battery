# PRD: Generic Tabular Regression Model Comparison

## Goal

CSV 기반 테이블형 데이터에서 회귀 문제를 정의하고, 여러 모델의 성능을 같은 프로토콜로 비교한다.

## Problem Definition

```text
X = 사용자가 지정한 feature columns
y = 사용자가 지정한 numeric target column
task = supervised regression
```

데이터셋은 고정하지 않는다. 조건은 다음뿐이다.

- CSV로 읽을 수 있는 테이블형 데이터
- 연속형 또는 수치형 회귀 타깃 존재
- 학습 시점에만 알 수 있는 누수 컬럼은 제외 가능

## Primary Interface

```powershell
.\.venv314\Scripts\python.exe train.py --config configs\example.json
```

또는:

```powershell
.\.venv314\Scripts\python.exe train.py `
  --csv "path\to\data.csv" `
  --target target `
  --features f1,f2,f3 `
  --model lightgbm
```

## Model Scope

Generic CSV mode supports:

- LightGBM
- CatBoost
- TabPFN v3

Other model wrappers can remain in the repository, but new generic support should be added only after the basic CSV contract is stable.

## Success Criteria

- same dataset/target/split can be run across multiple models
- result rows are appended to `results/tabular_regression_experiments.csv`
- feature columns and excluded columns are logged
- metrics include RMSE, MAE, MAPE, WAPE
- no dataset-specific assumptions are required in `train.py`
