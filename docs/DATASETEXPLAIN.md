# NASA Battery Dataset 설명

## 1. 데이터 출처

- Kaggle dataset: `patrickfleith/nasa-battery-dataset`
- 원천 계열: NASA Ames Prognostics Center of Excellence lithium-ion battery aging data
- 로컬 경로: `data/nasa_battery_raw/cleaned_dataset`
- 현재 프로젝트는 이전 데이터셋 사용을 중단하고, NASA battery cycle/degradation 데이터로 전환한다.

## 2. 로컬 파일 구조

```text
data/nasa_battery_raw/cleaned_dataset/
  metadata.csv
  data/
    00001.csv
    00002.csv
    ...
  extra_infos/
    README_*.txt
```

확인된 파일 규모:

| 항목 | 값 |
|---|---:|
| metadata rows | 7,565 |
| 개별 signal CSV | 7,565 |
| 전체 signal rows 합계 | 7,376,834 |
| battery cells | 34 |
| discharge records | 2,794 |
| charge records | 2,815 |
| impedance records | 1,956 |

## 3. 원본 스키마

`metadata.csv`:

| 컬럼 | 의미 | 사용 정책 |
|---|---|---|
| `type` | `charge`, `discharge`, `impedance` | discharge cycle 필터링에 사용 |
| `start_time` | 실험 시작 시간 | 1차 feature에서는 제외 |
| `ambient_temperature` | 주변 온도 | feature |
| `battery_id` | 배터리 셀 ID | group split key 및 categorical feature |
| `test_id` | 실험 순서 ID | feature |
| `uid` | 파일 고유 ID | metadata |
| `filename` | signal CSV 파일명 | metadata |
| `Capacity` | discharge capacity | target 원천 |
| `Re` | impedance ohmic resistance | 후속 impedance feature 후보 |
| `Rct` | impedance charge transfer resistance | 후속 impedance feature 후보 |

discharge signal CSV:

| 컬럼 | 의미 |
|---|---|
| `Voltage_measured` | 측정 전압 |
| `Current_measured` | 측정 전류 |
| `Temperature_measured` | 측정 온도 |
| `Current_load` | 부하 전류 |
| `Voltage_load` | 부하 전압 |
| `Time` | cycle 내부 시간 |

## 4. 1차 모델링 타깃

1차 타깃은 discharge cycle의 `capacity`이다.

```text
target = metadata.Capacity
unit = Ah 계열 capacity 값
```

`SOH`는 다음과 같이 계산한다.

```text
soh = capacity / first_capacity_of_same_battery
```

단, `soh`는 capacity에서 직접 계산되는 값이므로 기본 feature set에는 넣지 않는다. `discharge_health`는 분석용/상한 확인용 feature set으로만 사용한다.

## 5. 처리된 Cycle-Level 테이블

전처리 코드는 discharge record만 사용해 다음 파일을 생성한다.

```text
data/processed/nasa_cycle_level.csv
```

생성 규칙:

1. `metadata.csv`에서 `type == "discharge"`만 선택한다.
2. 각 row의 `filename`에 해당하는 signal CSV를 읽는다.
3. 전압/전류/온도/부하/시간 통계를 계산한다.
4. `battery_id`, `cycle_index`, `ambient_temperature`, `test_id`, `capacity`와 결합한다.
5. 배터리 셀 기준으로 정렬하여 저장한다.

## 6. Feature Sets

### `cycle_basic`

가장 단순한 cycle metadata feature set.

| 구분 | 컬럼 |
|---|---|
| numeric | `cycle_index`, `test_id`, `ambient_temperature` |
| categorical | `battery_id` |

### `discharge_summary`

1차 기본 feature set. discharge signal 전체를 cycle-level 통계로 요약한다.

추가 numeric features:

```text
sample_count
duration_sec
voltage_measured_first/last/min/max/mean/std
current_measured_first/last/min/max/mean/std
temperature_measured_first/last/min/max/mean/std
current_load_first/last/min/max/mean/std
voltage_load_first/last/min/max/mean/std
voltage_drop
mean_power_measured
integrated_abs_current
```

### `discharge_health`

`discharge_summary + soh`.

주의:

- `soh`는 target인 `capacity`에서 계산되므로 운영 성능 평가용 feature로 사용하지 않는다.
- 이 feature set은 leakage 상한 또는 sanity-check 전용이다.

## 7. Split 정책

NASA 데이터는 별도 train/validation 파일이 없다. 따라서 train/validation은 `battery_id` group holdout으로 나눈다.

```text
same battery_id cycles must not appear in both train and validation
```

이 정책은 같은 배터리의 인접 cycle이 train과 validation에 동시에 들어가 성능이 과대평가되는 문제를 막기 위한 필수 규칙이다.

## 8. 1차 실험 기준

- 기본 모델: LightGBM
- 기본 feature set: `discharge_summary`
- 기본 split: `battery_id_group_holdout`
- 주요 metric: MAE, RMSE, WAPE, SMAPE
- raw MAPE는 target이 작을 때 불안정하므로 보조 지표로만 기록한다.

## 9. 이전 데이터셋 폐기 정책

이전 데이터셋은 현재 프로젝트의 active dataset에서 제외한다.

이전 실험은 보존된 결과 해석용으로만 취급하며, 이후 모델 실험, 문서, leaderboard는 NASA cycle-level dataset을 기준으로 갱신한다.
