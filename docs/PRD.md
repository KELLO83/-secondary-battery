# PRD: NASA Battery Cycle-Level Capacity Prediction

## 1. Objective

NASA battery degradation data를 사용해 cycle-level tabular 모델을 구축하고, 배터리 capacity/SOH 예측에 실질적으로 유효한 ML/AI 모델군을 선정한다.

이전 데이터셋 실험은 active scope에서 제외한다.

## 2. Dataset

- Dataset: Kaggle `patrickfleith/nasa-battery-dataset`
- Raw path: `data/nasa_battery_raw/cleaned_dataset`
- Processed table: `data/processed/nasa_cycle_level.csv`
- Raw signal rows: 약 7.38M
- Cycle-level discharge rows: 약 2.8K
- Battery cells: 34

## 3. Prediction Task

1차 task:

```text
X = discharge cycle metadata + discharge signal summary
y = capacity
```

후속 task:

```text
SOH estimation
RUL prediction
early-cycle capacity forecasting
```

## 4. Feature Policy

기본 feature set은 `discharge_summary`이다.

| Feature set | 설명 | 1차 사용 |
|---|---|---|
| `cycle_basic` | `cycle_index`, `test_id`, `ambient_temperature`, `battery_id` | baseline |
| `discharge_summary` | discharge signal 통계 포함 | default |
| `discharge_health` | `soh` 포함 | sanity-check only |

`discharge_health`는 target-derived feature를 포함하므로 정상 leaderboard에 넣지 않는다.

## 5. Split Policy

모든 정상 실험은 battery-level holdout을 사용한다.

```text
train battery_id set ∩ validation battery_id set = empty
```

이유:

- 같은 battery의 cycle은 강하게 상관되어 있다.
- random row split은 인접 cycle 정보 누수로 validation 성능을 과대평가한다.

## 6. Model Roadmap

Phase 1:

1. LightGBM
2. CatBoost

Phase 2:

1. RealMLP
2. TabM
3. NODE
4. TabR

Phase 3:

1. FT-Transformer
2. TabTransformer
3. TabNet

Phase 4:

1. TabICLv2
2. TabPFN v3

단, cycle-level row 수가 약 2.8K로 작기 때문에 대형 transformer는 과적합 가능성이 높다. NASA에서는 LightGBM/CatBoost/TabM/NODE/TabICLv2를 우선한다.

## 7. Success Metrics

Primary:

```text
MAE
RMSE
WAPE
```

Secondary:

```text
SMAPE
filtered MAPE
raw MAPE
```

1차 통과 기준:

```text
WAPE <= 10%
```

권장 목표:

```text
WAPE <= 5%
```

## 8. Artifacts

필수 산출물:

```text
data/processed/nasa_cycle_level.csv
results/experiments.csv
results/logs/*.log
```

모델 저장은 성능 검증 후 별도 artifact 정책으로 추가한다.

## 9. Operational Rules

- 한 번의 `train.py` 실행은 하나의 모델 실험만 수행한다.
- 여러 모델을 자동으로 연속 실행하는 sweep은 기본 금지한다.
- 실험 로그는 파일과 터미널에 모두 남긴다.
- LightGBM 1차 실험 후 CatBoost GPU, TabM/NODE 순서로 확장한다.
