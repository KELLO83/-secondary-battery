# DATA_CONTRACT: 2차전지 데이터 입력 계약

## 1. 목적

본 문서는 `docs/DATASETEXPLAIN.md`의 데이터 설명을 코드에서 검증 가능한 입력 계약으로 정리한다.

데이터 로더, 전처리 파이프라인, 예측 API는 본 문서의 규칙을 기준으로 입력 데이터를 검증한다.

공식 원천 데이터:

```text
AI Hub Dataset Sn: 71869
https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71869
```

## 2. 필수 칼럼

본 계약은 AI Hub에서 제공하는 preprocessing 이후 학습 기준 칼럼을 대상으로 한다.

### 2.1 Target

```text
remain_capacity
```

### 2.2 Numeric Features

```text
voltage_range(V)_min
voltage_range(V)_max
```

### 2.3 Categorical Features

```text
material_structure
synthesis_method
Li_source
Ni_source
Co_source
Mn_source
electrolyte
separator
counter_electrode
```

### 2.4 Metadata Columns

있을 수 있지만 학습 feature로 사용하지 않는다.

```text
material_id
chemical_formula
DOI
journal_name
Class
Unnamed: 0
```

추가로 다음 성격의 칼럼이 존재하면 학습 feature에서 제외한다.

```text
수집시간
수집종료시간
파일생성시간
작업자
원천 시스템 ID
다운로드/적재 로그성 칼럼
```

### 2.5 Derived Metadata

파일명에서 다음 metadata를 파생할 수 있다.

```text
source_family: LFP | NCA | NCM | Others
```

규칙:

- `source_family`는 학습 feature로 사용하지 않는다.
- `source_family`는 계열별 metric, OOD 분석, drift monitoring, 리포트 필터링에만 사용한다.
- API 예측 요청에서 사용자가 `source_family`를 직접 입력하더라도 모델 입력 feature에는 포함하지 않는다.

## 2.6 Feature Set 계약

모델 학습 코드는 단계별 feature set을 지원한다.

### `core_11`

1차 모델 개발 기준 feature set이다.

사용 칼럼:

```text
material_structure
synthesis_method
Li_source
Ni_source
Co_source
Mn_source
electrolyte
separator
counter_electrode
voltage_range(V)_min
voltage_range(V)_max
```

제외 칼럼:

```text
material_id
remain_capacity
source_family
core_11 외 모든 확장 칼럼
```

### `design_15`, 후순위 확장 모델

```text
core_11
sintering_T1(C)
sintering_t1(h)
measurement_T(C)
C-rate
```

추가 입력 4개:

| 추가 칼럼 | 타입 | 의미 |
| :--- | :--- | :--- |
| `sintering_T1(C)` | numeric | 1차 소결 또는 열처리 온도 |
| `sintering_t1(h)` | numeric | 1차 소결 또는 열처리 시간 |
| `measurement_T(C)` | numeric | 성능 측정 온도 |
| `C-rate` | numeric | 충방전 속도 조건 |

### `chem_22`, 후순위 확장 모델

```text
design_15
Li_fraction
Ni_fraction
Mn_fraction
Co_fraction
dopant_fraction
active_proportion
binder_proportion
```

추가 입력 7개:

| 추가 칼럼 | 타입 | 의미 |
| :--- | :--- | :--- |
| `Li_fraction` | numeric | 리튬 조성 비율 |
| `Ni_fraction` | numeric | 니켈 조성 비율 |
| `Mn_fraction` | numeric | 망간 조성 비율 |
| `Co_fraction` | numeric | 코발트 조성 비율 |
| `dopant_fraction` | numeric | 도핑 원소 조성 비율 |
| `active_proportion` | numeric | 전극 내 활물질 비율 |
| `binder_proportion` | numeric | 전극 내 바인더 비율 |

### `official`, 후속 ablation/benchmark 후보

AI Hub 공식 학습 기준 feature set이다.

사용 칼럼:

```text
material_structure
synthesis_method
sintering_T1(C)
sintering_t1(h)
Li_source
Co_source
Mn_source
Ni_source
electrolyte
counter_electrode
separator
measurement_T(C)
Li_fraction
Ni_fraction
Mn_fraction
Co_fraction
dopant_fraction
active_proportion
binder_proportion
particle_size(um)
C-rate
discharge_capacity (mAh/g)
Strain
state_of_charge
space_group_symbol
length_a
length_b
length_c
angle_alpha
angle_beta
angle_gamma
volume
density
interlayer_dist
energy
tm_o_bond_length
perc_barrier_1d
perc_barrier_2d
perc_radius_1d
perc_radius_2d
max_packing_eff
chemical_ordering
struct_hetero_bond
struct_hetero_cell
voltage_range(V)_min
voltage_range(V)_max
```

제외 칼럼:

```text
material_id
remain_capacity
source_family
```

기본값:

```text
1차 모델 개발: core_11
agent/design-oriented experiment: core_11
후속 성능 개선 후보: design_15, chem_22
후속 ablation/benchmark: official
```

중요:

- 1차 모델 개발에서는 `core_11`만 사용한다.
- `design_15`, `chem_22`, `official`은 `core_11`에서 우수한 모델군을 찾은 뒤 별도 모델로 학습/평가한다.
- 48개 raw column 전체를 자동 feature로 넣는 학습은 금지한다.

## 3. Feature 계약

### 3.1 Numeric Feature

규칙:

- float로 변환 가능해야 한다.
- 결측치는 허용하지 않는 것을 기본으로 한다.
- `voltage_range(V)_min <= voltage_range(V)_max`를 만족해야 한다.
- 학습 데이터 범위를 벗어나는 값은 OOD 후보로 표시한다.

### 3.2 Categorical Feature

규칙:

- 문자열 또는 category dtype으로 처리한다.
- 결측치는 `__MISSING__` category로 치환할 수 있다.
- validation/test/serving에서 처음 등장한 category는 `__UNKNOWN__`으로 처리한다.
- high-cardinality 여부는 전처리 로그에 기록한다.

## 4. Target 계약

`remain_capacity` 규칙:

- float로 변환 가능해야 한다.
- 결측치는 학습에서 제외한다.
- 기본 필터링 범위:

```text
0 <= remain_capacity < 1000
```

주의:

- target filter 적용 전후 row 수를 기록한다.
- target이 0에 가까운 sample은 MAPE 불안정성 분석 대상이다.

## 5. 학습 제외 규칙

다음 칼럼은 모델 feature에 포함하지 않는다.

```text
material_id
chemical_formula
DOI
journal_name
Class
Unnamed: 0
source_family
```

단, 다음 용도로는 사용할 수 있다.

- group split key
- leakage 검사
- 원천 문헌 추적
- 서비스 화면 표시 metadata

## 6. API 입력 매핑

서비스 API에서는 JSON 친화적인 필드명을 사용할 수 있다.

필수 매핑:

```text
voltage_range_V_min -> voltage_range(V)_min
voltage_range_V_max -> voltage_range(V)_max
```

그 외 categorical feature는 원본 칼럼명과 동일한 이름을 사용한다.

## 7. Validation 실패 처리

데이터 로딩 또는 API 요청에서 계약 위반이 발생하면 명시적 오류를 반환한다.

오류 유형:

```text
missing_required_column
invalid_numeric_value
invalid_target_value
invalid_voltage_range
unknown_feature
metadata_used_as_feature
```

## 8. 코드 반영 위치

권장 구현 위치:

```text
src/schema.py
src/data/loader.py
src/data/preprocessing.py
backend/app/schemas/prediction.py
backend/app/ml/preprocessing.py
```

## 9. 변경 관리

feature, target, metadata 정의가 바뀌면 다음 문서를 함께 확인한다.

- `docs/DATASETEXPLAIN.md`
- `docs/SYSTEM_ARCH.md`
- `docs/SERVICE_ARCH.md`
- `docs/XAI_MONITORING.md`
