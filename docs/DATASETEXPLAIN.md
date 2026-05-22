# 고품질 연구개발용 리튬 이온 이차 전지 데이터셋 명세서
(AI Hub Dataset Sn: 71869)

* **원천 데이터 URL**: https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71869

본 문서는 리튬 이온 이차전지의 신소재 설계 및 성능 예측 AI 에이전트 시스템 구축을 위한 **'고품질 연구개발용 리튬 이온 이차 전지 데이터셋'**의 상세 형태와 스키마를 설명합니다.

---

## 1. 데이터셋 개요

* **데이터셋명**: 고품질 연구개발용 리튬 이온 이차 전지 데이터
* **구축년도 / 갱신년월**: 2024년 / 2025년 6월
* **공식 데이터 출처**: AI Hub 데이터셋 `71869`
* **데이터 크기**: **1.40 GB**
* **데이터 규모**: 총 **16,325,872 건** (약 1,632만 건의 대용량 레코드)
  * **학습(Training)**: 13,060,696 건 (80%)
  * **검증(Validation)**: 1,632,587 건 (10%)
  * **시험(Test)**: 1,632,589 건 (10%)
* **데이터 형식**: 테이블 구조의 정형 데이터 (`.csv` 파일)
* **원천데이터 수집 출처**: 소재과학(Materials Science) 분야의 권위 있는 저널 논문 문헌들로부터 대형 언어 모델(LLM) 등을 활용해 전지 구조, 소재 배합, 실험 결과 데이터를 수집·통합하여 구축.

---

## 1.1 현재 워크스페이스 CSV 파일 배치

AI Hub 원천 zip 파일은 압축 해제 후 코드에서 다루기 쉬운 경로로 정리하였다.

현재 학습/검증 CSV 파일은 다음 위치에 둔다.

### Training CSV

| 파일 경로 | 데이터 계열 | 용도 |
| :--- | :--- | :--- |
| `Training/LFP_train_dataset.csv` | LFP (리튬-철-인) | 학습 데이터 |
| `Training/NCA_train_dataset.csv` | NCA (리튬-니켈-코발트-알루미늄) | 학습 데이터 |
| `Training/NCM_train_dataset.csv` | NCM (리튬-니켈-코발트-망간) | 학습 데이터 |
| `Training/others_train_dataset.csv` | Others (LNMO, Lithium rich 등) | 학습 데이터 |

### Validation CSV

| 파일 경로 | 데이터 계열 | 용도 |
| :--- | :--- | :--- |
| `Validation/LFP_val_dataset.csv` | LFP (리튬-철-인) | 검증 데이터 |
| `Validation/NCA_val_dataset.csv` | NCA (리튬-니켈-코발트-알루미늄) | 검증 데이터 |
| `Validation/NCM_val_dataset.csv` | NCM (리튬-니켈-코발트-망간) | 검증 데이터 |
| `Validation/others_val_dataset.csv` | Others (LNMO, Lithium rich 등) | 검증 데이터 |

주의:

* 원천 zip 파일은 압축 해제 후 삭제하였다.
* 기존 AI Hub 압축 해제 폴더의 한글/괄호 포함 경로는 코드 경로 단순화를 위해 사용하지 않는다.
* 학습 코드는 위 8개 CSV 경로를 기준으로 데이터를 로딩한다.
* 현재 8개 CSV는 모두 동일하게 `remain_capacity` 포함 총 48개 칼럼을 가진 확장 스키마이다.
* 기본 모델링 전략은 4개 계열 CSV를 병합해 하나의 통합 모델을 학습하는 것이다.
* 파일명에서 `source_family` (`LFP`, `NCA`, `NCM`, `Others`)를 파생할 수 있으나, 이는 학습 feature가 아니라 평가/분석/OOD 모니터링용 metadata로만 사용한다.
* 모델 훈련에서는 48개 칼럼의 장황한 도메인 설명보다 feature/target/metadata 구분을 우선한다.
* 수집시간, 수집종료시간, 파일생성시간, 작업자, 원천 시스템 ID 같은 운영/수집 메타데이터가 추가로 존재할 경우 학습 feature로 사용하지 않는다.
* 시험(Test) 데이터가 별도 제공되는 경우 `Test/` 또는 `Testing/` 아래 같은 방식으로 정리한다.

### 통합 모델 전략

본 프로젝트는 초기 실험에서 LFP, NCA, NCM, Others 계열별 specialized model을 따로 학습하지 않는다.

기본 전략:

```text
Training/*.csv 전체 병합 -> 통합 모델 1개 학습
Validation/*.csv 전체 병합 -> 전체 metric 및 source_family별 metric 산출
```

이유:

* 모델 후보, sample size, split, hyperparameter 조합이 많아 계열별 모델까지 학습하면 실험 수가 과도하게 증가한다.
* 통합 모델은 계열 간 공유되는 소재/공정/전압 패턴을 함께 학습할 수 있다.
* 계열별 성능 문제는 모델을 4개로 쪼개기보다 먼저 `source_family`별 metric, OOD, error 분석으로 확인한다.

후속 조건:

* 특정 계열에서 통합 모델의 group split 성능이 지속적으로 낮고, 데이터 수가 충분한 경우에만 Phase 5 이후 계열별 specialized model을 검토한다.

---

## 2. 데이터 분석 및 모델링 성격

* **예측 문제 정의**: **회귀 (Regression) 문제**
  * 본 데이터셋은 배터리의 오류나 불량을 판정하는 분류(Classification)용 데이터가 아닙니다.
  * 다양한 소재 조합과 제어 변수 조건 속에서 배터리가 발현할 **실제 '잔류 방전 용량(Remain Capacity)'의 절대 수치**를 도출하는 연속값 예측 모델을 위한 데이터셋입니다.
* **벤치마크 모델 성능**: **MAPE (Mean Absolute Percentage Error) 2.969%**
  * AI 예측값과 실제 실험 측정값의 평균 오차가 **약 2.97%**로, 약 **97% 이상의 대단히 정교한 예측 정확도**를 나타냅니다.

---

## 3. 데이터 스키마 및 피처 명세

학습 데이터셋은 원래 핵심 설명 기준으로 **11개의 주요 독립변수(X)**와 **1개의 종속변수(Y)**를 중심으로 설명되었으나, 현재 워크스페이스 CSV에는 소결 조건, 조성비, 구조/물성 계산값 등 추가 피처가 포함된 **48개 칼럼 확장 스키마**가 들어 있다.

코드에서는 단계별 feature set을 지원한다.

```text
core_11: 1차 모델 개발 기준. 핵심 11개 입력 변수만 사용
design_15: 후순위 확장 모델. core_11 + 합성/측정 조건 4개
chem_22: 후순위 확장 모델. design_15 + 조성/전극 비율 7개
official: AI Hub 공식 학습 기준 feature. 후속 ablation/benchmark 후보
```

1차 모델 개발의 기본 feature set은 `core_11`이다. `design_15`, `chem_22`, `official`은 `core_11`에서 우수한 모델군을 찾은 뒤 별도 모델로 추가 실험한다.

모델 훈련 기준:

```text
target:
  remain_capacity

metadata / 학습 제외:
  material_id
  source_family
  AI Hub 공식 학습 제외 column: chemical_formula, DOI, Unnamed: 0, Class, journal_name, 존재 시
  수집시간/수집종료시간 등 운영 메타데이터, 존재 시

phase 1 features:
  core_11 11개 칼럼만 사용

later features:
  design_15, chem_22, official은 후순위 별도 모델에서만 사용
```

### Feature Set 정의

본 프로젝트는 목적에 따라 feature set을 구분한다. 1차 개발과 leaderboard는 `core_11`을 기준으로 만든다.

#### 1. `core_11` feature set: 1차 모델 개발 기준

1차 모델은 다음 11개 칼럼만 훈련 feature로 사용한다.

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

사용 목적:

* 1차 ML/AI 모델 개발
* 연구원이 직접 입력 가능한 recipe 기반 예측
* AI 에이전트의 `capacity_predictor_tool` 기본 입력 schema
* 예측 시점에 알기 어려운 후처리/계산/측정 feature 배제

`core_11` numeric features:

```text
voltage_range(V)_min
voltage_range(V)_max
```

`core_11` categorical features:

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

#### 2. 후속 feature set 후보

1차 모델에서 우수한 모델군을 찾은 뒤, 성능 개선이 필요하면 다음 feature set을 별도 모델로 비교한다. `design_15`와 `chem_22`는 `core_11` 모델의 입력 스키마를 바꾸는 별도 모델이며, 1차 기본 실험에 섞지 않는다.

`design_15`:

```text
core_11
sintering_T1(C)
sintering_t1(h)
measurement_T(C)
C-rate
```

`design_15`에서 추가로 입력받는 4개 feature:

| 추가 칼럼 | 타입 | 의미 | 추가 목적 |
| :--- | :--- | :--- | :--- |
| `sintering_T1(C)` | 수치형 | 1차 소결 또는 열처리 온도 | 합성 공정 강도와 결정화/입자 성장 조건 반영 |
| `sintering_t1(h)` | 수치형 | 1차 소결 또는 열처리 시간 | 열처리 지속시간에 따른 구조 안정화 효과 반영 |
| `measurement_T(C)` | 수치형 | 성능 측정 온도 | 측정 환경 차이가 용량에 주는 영향 보정 |
| `C-rate` | 수치형 | 충방전 속도 조건 | 부하 조건에 따른 용량 저하/회복 패턴 반영 |

`chem_22`:

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

`chem_22`에서 추가로 입력받는 7개 feature:

| 추가 칼럼 | 타입 | 의미 | 추가 목적 |
| :--- | :--- | :--- | :--- |
| `Li_fraction` | 수치형 | 리튬 조성 비율 | 리튬 과잉/부족 조성이 용량에 주는 영향 반영 |
| `Ni_fraction` | 수치형 | 니켈 조성 비율 | 고용량에 기여하는 Ni 함량 효과 반영 |
| `Mn_fraction` | 수치형 | 망간 조성 비율 | 구조 안정성 및 전이금속 조성 효과 반영 |
| `Co_fraction` | 수치형 | 코발트 조성 비율 | 층상 구조 안정화 및 전극 성능 영향 반영 |
| `dopant_fraction` | 수치형 | 도핑 원소 조성 비율 | 미량 첨가 원소의 성능 보정 효과 반영 |
| `active_proportion` | 수치형 | 전극 내 활물질 비율 | 실제 전극 구성에서 용량 발현 비중 반영 |
| `binder_proportion` | 수치형 | 전극 내 바인더 비율 | 전극 조성 변화가 유효 용량에 주는 영향 반영 |

#### 3. `official` feature set: 후속 ablation/benchmark 후보

AI Hub 공식 페이지의 preprocessing 설명은 학습에 사용되지 않는 column을 다음과 같이 명시한다.

```text
material_id
chemical_formula
DOI
Unnamed: 0
Class
journal_name
```

현재 워크스페이스 CSV에는 위 학습 제외 column 중 `material_id`만 남아 있다.

`official` feature set은 현재 CSV에서 `material_id`, `remain_capacity`, `source_family`를 제외한 공식 학습 입력 칼럼이다.

사용 목적:

* AI Hub 공식 학습 feature 기준 benchmark 재현
* core_11/design_15/chem_22 대비 성능 차이 확인
* leakage 또는 예측 시점 가용성 진단

`official` numeric features:

```text
sintering_T1(C)
sintering_t1(h)
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

`official` categorical features:

```text
material_structure
synthesis_method
Li_source
Co_source
Mn_source
Ni_source
electrolyte
counter_electrode
separator
space_group_symbol
```

#### 학습 제외 칼럼

```text
material_id
remain_capacity
source_family
chemical_formula, DOI, Unnamed: 0, Class, journal_name, 존재 시
수집시간/수집종료시간 등 운영 메타데이터, 존재 시
```

주의:

* `remain_capacity`는 target이므로 feature로 사용하지 않는다.
* `material_id`는 식별자이므로 feature로 사용하지 않는다.
* `source_family`는 평가/분석/OOD 모니터링용 metadata이며 feature로 사용하지 않는다.
* 1차 모델에서는 `core_11` 외 모든 칼럼을 학습 feature로 사용하지 않는다.
* `discharge_capacity (mAh/g)`, `state_of_charge` 등 일부 official feature는 예측 시점에 실제로 알 수 있는지 별도 검토가 필요하다. 따라서 `official`은 1차 모델이 아니라 후속 ablation/benchmark로만 사용한다.

### ① 독립변수 (Input Features, X)

| 변수구분 | 칼럼명 (Column Name) | 변수 타입 | 데이터 예시 및 설명 |
| :--- | :--- | :--- | :--- |
| **작동 조건** | `voltage_range(V)_min` | 수치형 (Float) | 충방전 테스트 시 작동 **최소 전압** (예: `2.0`, `2.5`) |
| **작동 조건** | `voltage_range(V)_max` | 수치형 (Float) | 충방전 테스트 시 작동 **최대 전압** (예: `4.2`, `4.3`) |
| **결정 구조** | `material_structure` | 범주형 (String) | 양극재 소재의 결정 구조 (예: `Layered`, `Spinel`, `Olivine` 등) |
| **합성 공정** | `synthesis_method` | 범주형 (String) | 소재를 합성한 화학적 공정 기법 (예: `Solid-state`, `Co-precipitation` 등) |
| **원료 물질** | `Li_source` | 범주형 (String) | 리튬 공급원 전구체 물질 종류 (예: `Li2CO3`, `LiOH`, `CH3COOLi` 등) |
| **원료 물질** | `Ni_source` | 범주형 (String) | 니켈 공급원 전구체 물질 종류 (예: `Ni(NO3)2`, `NiSO4`, `NiO` 등) |
| **원료 물질** | `Co_source` | 범주형 (String) | 코발트 공급원 전구체 물질 종류 (예: `Co(NO3)2`, `Co3O4` 등) |
| **원료 물질** | `Mn_source` | 범주형 (String) | 망간 공급원 전구체 물질 종류 (예: `Mn(NO3)2`, `MnCO3`, `MnO2` 등) |
| **셀 구성품** | `electrolyte` | 범주형 (String) | 사용된 전해질 용액의 화학적 조합 종류 |
| **셀 구성품** | `separator` | 범주형 (String) | 양극과 음극 사이 쇼트를 방지하는 분리막 재질 종류 |
| **셀 구성품** | `counter_electrode` | 범주형 (String) | 성능 측정을 위한 기준 상대전극(대극) 종류 (예: `Li-metal`) |

### ② 종속변수 (Target Label, Y)

| 칼럼명 (Column Name) | 변수 타입 | 데이터 설명 |
| :--- | :--- | :--- |
| **`remain_capacity`** | 수치형 (Float) | AI 모델이 예측해야 하는 배터리의 **잔류 방전 용량** (절대 수치 값) |

### ③ 학습에서 배제되는 메타데이터 칼럼
학습 모델에 직접 입력하지 않고, 데이터 조회 및 원천 문헌 확인용으로만 활용되는 메타데이터성 칼럼입니다. 전처리 단계에서 학습 세트로부터 제거(Drop)해야 합니다.
* `material_id` (소재 식별 번호)
* `chemical_formula` (화학식 텍스트)
* `DOI` (원천 논문 디지털 객체 식별자)
* `journal_name` (해당 논문이 게재된 저널명)
* `Class` / `Unnamed: 0` (단순 행 인덱스 및 분류 마크)

---

## 4. 권장 전처리 및 파이프라인 가이드라인

1. **결측치 및 극단치 제거**: 
   * 데이터 전처리 시 불필요한 메타 칼럼을 제거한 후, 타겟값인 `remain_capacity`가 지나치게 튀는 값이나 비정상적인 데이터(예: `1000` 이상이거나 음수인 데이터 등)를 사전에 필터링합니다.
2. **범주형 변수 처리 (One-Hot Encoding)**:
   * 9개의 범주형 칼럼들은 AI 모델(신경망 등)이 읽을 수 있도록 원-핫 인코딩 처리를 합니다. 
   * 전체 카테고리의 고유 값 수에 따라 모델의 입력 차원(Input Dimension)이 결정됩니다.
3. **수치형 변수 스케일링 (Standard Scaling)**:
   * 수치형인 `voltage_range_min`, `voltage_range_max` 그리고 최종 라벨인 `remain_capacity`는 학습 안정성을 증대시키기 위해 정규화 스케일러(StandardScaler 등)를 사용해 정규화합니다.
