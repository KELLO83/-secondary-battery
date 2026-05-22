# Evaluation Protocol

## Scope

본 문서는 NASA battery cycle-level capacity prediction 실험의 평가 규칙을 정의한다.

## Split

기본 split은 `battery_id_group_holdout`이다.

금지:

- cycle row 단위 random split
- 같은 `battery_id`가 train/validation에 동시에 들어가는 split

허용:

- `GroupShuffleSplit(groups=battery_id)`
- leave-one-battery-out
- battery family/temperature 조건을 고려한 후속 group split

## Metrics

Primary:

```text
MAE = mean(abs(y_true - y_pred))
RMSE = sqrt(mean((y_true - y_pred)^2))
WAPE = sum(abs(error)) / sum(abs(y_true)) * 100
```

Secondary:

```text
SMAPE
filtered MAPE
raw MAPE
```

raw MAPE는 capacity가 작을 때 불안정하므로 모델 선정 기준으로 단독 사용하지 않는다.
filtered MAPE는 NASA capacity 스케일에 맞춰 `abs(capacity) > 0.1` 기준으로 계산한다.

## Feature Set Reporting

모든 결과 row에는 feature set을 명시한다.

정상 비교 가능:

```text
cycle_basic
discharge_summary
```

정상 leaderboard 제외:

```text
discharge_health
```

`discharge_health`는 `soh = capacity / first_capacity`를 포함하므로 target-derived sanity-check로만 사용한다.

## Minimum Acceptance

초기 모델 실험에서 다음 기준을 목표로 한다.

| 기준 | 값 |
|---|---:|
| 최소 통과 | WAPE <= 10% |
| 권장 목표 | WAPE <= 5% |

대형 모델을 적용해도 WAPE가 10% 아래로 내려가지 않으면 feature engineering 또는 task 재정의가 필요하다.

## Logging

각 실험은 다음을 남긴다.

```text
model_name
feature_set
split_type
train rows
validation rows
train_time_sec
predict_time_sec
MAE/RMSE/WAPE/SMAPE/MAPE
source_family_metrics by battery_id
```
