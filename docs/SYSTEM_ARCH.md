# System Architecture

## Repository Layout

```text
data/
  nasa_battery_raw/
    cleaned_dataset/
      metadata.csv
      data/*.csv
  processed/
    nasa_cycle_level.csv

ml/
  src/
    data/
      nasa_cycle_builder.py
      loader.py
      preprocessing.py
    models/
      gbdt/
      neural/
      transformer/
      foundation/
    eval/
    experiments/
  tests/

results/
  experiments.csv
  logs/*.log
```

## Data Flow

```text
metadata.csv
  + data/*.csv discharge signal files
  -> ml/src/data/nasa_cycle_builder.py
  -> data/processed/nasa_cycle_level.csv
  -> battery_id group split
  -> preprocessing
  -> model.fit()
  -> metrics/logs/results
```

## Core Interfaces

### Dataset builder

```python
build_nasa_cycle_level_dataset(force=False) -> pd.DataFrame
```

역할:

- discharge row만 선택
- signal CSV 통계 생성
- cycle-level table 저장

### Loader

```python
sample_integrated_split(split, sample_size, seed, feature_set) -> pd.DataFrame
```

역할:

- NASA processed table 로드
- `battery_id` group holdout split 수행
- 필요 시 sample 적용

### Model

모든 모델은 `BaseModel` 인터페이스를 따른다.

```python
fit(X_train, y_train, X_valid, y_valid)
predict(X) -> np.ndarray
```

## Feature Sets

```text
cycle_basic
discharge_summary
discharge_health
```

기본값은 `discharge_summary`이다.

## Experiment Runner

`train.py`는 한 번에 하나의 모델만 실행한다.

예:

```powershell
.\.venv314\Scripts\python.exe train.py --model lightgbm --feature-set discharge_summary --full-data --valid-full-data
```

## Logging

중요 단계는 `INFO` 로그로 남긴다.

- 데이터 로딩
- feature/target 준비
- 모델 생성
- 학습 시작/종료
- 추론 시작/종료
- metric 저장

훈련 진행률이 가능한 모델은 진행바를 표시한다.
