# NASA Data Contract

## Active Dataset

현재 active dataset은 Kaggle `patrickfleith/nasa-battery-dataset`이다.

```text
raw root: data/nasa_battery_raw/cleaned_dataset
processed table: data/processed/nasa_cycle_level.csv
target: capacity
group key: battery_id
```

이전 데이터셋은 사용하지 않는다.

## Required Raw Files

```text
data/nasa_battery_raw/cleaned_dataset/metadata.csv
data/nasa_battery_raw/cleaned_dataset/data/*.csv
```

`metadata.csv`에는 최소 다음 컬럼이 필요하다.

```text
type
ambient_temperature
battery_id
test_id
uid
filename
Capacity
```

discharge signal CSV에는 최소 다음 컬럼이 필요하다.

```text
Voltage_measured
Current_measured
Temperature_measured
Current_load
Voltage_load
Time
```

## Processed Table Contract

`data/processed/nasa_cycle_level.csv`는 discharge cycle 1개를 row 1개로 표현한다.

필수 metadata/target:

```text
battery_id
test_id
uid
filename
ambient_temperature
cycle_index
capacity
soh
```

필수 signal summary:

```text
sample_count
duration_sec
voltage_measured_first
voltage_measured_last
voltage_measured_min
voltage_measured_max
voltage_measured_mean
voltage_measured_std
current_measured_first
current_measured_last
current_measured_min
current_measured_max
current_measured_mean
current_measured_std
temperature_measured_first
temperature_measured_last
temperature_measured_min
temperature_measured_max
temperature_measured_mean
temperature_measured_std
current_load_first
current_load_last
current_load_min
current_load_max
current_load_mean
current_load_std
voltage_load_first
voltage_load_last
voltage_load_min
voltage_load_max
voltage_load_mean
voltage_load_std
voltage_drop
mean_power_measured
integrated_abs_current
```

## Feature Sets

| Feature set | 용도 | 정상 leaderboard 사용 |
|---|---|---|
| `cycle_basic` | cycle index/온도만 쓰는 최소 기준선 | 가능 |
| `discharge_summary` | 1차 기본 모델 | 가능 |
| `discharge_health` | `soh` 포함 sanity-check/상한 확인 | 불가 |

`discharge_health`는 target-derived `soh`를 포함하므로 운영 성능 비교에는 사용하지 않는다.

## Split Contract

모든 정상 실험은 `battery_id` group holdout을 사용한다.

금지:

```text
same battery_id in train and validation
random row split over cycles
```

허용:

```text
GroupShuffleSplit(groups=battery_id)
leave-one-battery-out style validation
```

## Metric Contract

주요 지표:

```text
MAE
RMSE
WAPE
SMAPE
```

보조 지표:

```text
raw MAPE
filtered MAPE
```

raw MAPE는 capacity가 작거나 0에 가까운 row에서 불안정할 수 있으므로 최종 선정 기준으로 사용하지 않는다.
