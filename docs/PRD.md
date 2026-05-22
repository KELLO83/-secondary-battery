# PRD: 2차전지 Tabular ML/Deep Learning/SOTA Foundation Model 실험

## 1. 목적

본 문서는 `docs/DATASETEXPLAIN.md`에 정의된 2차전지 정형 데이터셋을 대상으로, 기존 머신러닝 모델과 최신 정형 데이터 딥러닝/파운데이션 모델을 체계적으로 비교하기 위한 실험 요구사항을 정의한다.

핵심 목표는 다음과 같다.

- `LightGBM`, `CatBoost` 기반의 강한 실무 baseline 확보
- `FT-Transformer`, `TabTransformer`, `DCN-V2`, `NODE`, `TabM`, `TabR`, `RealMLP` 등 정형 데이터 딥러닝 모델 비교
- `TabPFN-3/latest`, `TabICLv2` 등 최신 Tabular Foundation Model을 샘플링 데이터에서 검증
- `AutoGluon/Mitra`는 단일 모델이 아니라 ceiling benchmark로 분리 검증
- random split과 group split 성능 차이를 통해 데이터 누수 가능성 평가
- 최종적으로 2차전지 `remain_capacity` 예측에 가장 실용적인 모델군 선정

## 2. 데이터 개요

기준 문서: `docs/DATASETEXPLAIN.md`

- 공식 데이터 출처: AI Hub Dataset Sn `71869`
- URL: https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71869
- 데이터 규모: 총 16,325,872건
- 데이터 크기: 약 1.40GB
- Training 데이터 규모: 총 13,060,696건
  - LFP: 131,790건, 약 1%
  - NCA: 769,225건, 약 6%
  - NCM: 8,750,613건, 약 67%
  - Others: 3,409,068건, 약 26%
- 문제 유형: 회귀
- 타겟 변수: `remain_capacity`
- 1차 입력 변수: `core_11` 총 11개
  - 수치형 2개
    - `voltage_range(V)_min`
    - `voltage_range(V)_max`
  - 범주형 9개
    - `material_structure`
    - `synthesis_method`
    - `Li_source`
    - `Ni_source`
    - `Co_source`
    - `Mn_source`
    - `electrolyte`
    - `separator`
    - `counter_electrode`
- 후순위 확장 입력 변수
  - `design_15`: `core_11` + `sintering_T1(C)`, `sintering_t1(h)`, `measurement_T(C)`, `C-rate`
  - `chem_22`: `design_15` + `Li_fraction`, `Ni_fraction`, `Mn_fraction`, `Co_fraction`, `dopant_fraction`, `active_proportion`, `binder_proportion`
  - `chem_derived`: `chem_22` + domain-derived leakage-safe features
- target-derived leakage feature, 정상 모델 학습 제외
  - `discharge_capacity (mAh/g)`
  - `state_of_charge`
- 학습 제외 메타데이터 후보
  - `material_id`
  - `chemical_formula`
  - `DOI`
  - `journal_name`
  - `Class`
  - `Unnamed: 0`

## 3. 성공 기준

### 3.1 모델 성능 기준

모든 모델은 동일한 split과 동일한 전처리 정책 아래에서 비교한다.

주요 평가 지표:

- RMSE
- MAE
- WAPE
- SMAPE
- filtered MAPE
- raw MAPE, diagnostic only
- 학습 시간
- 추론 시간
- peak memory 또는 GPU memory 사용량

Metric 우선순위:

- Primary: RMSE, MAE, WAPE
- Secondary: SMAPE, `abs(remain_capacity) > 5` 기준 filtered MAPE
- Diagnostic only: raw MAPE

raw MAPE는 `remain_capacity=0` 또는 0에 가까운 row에서 폭발하므로 최종 모델 선정 기준으로 사용하지 않는다.

### 3.1.1 운영 투입 마지노선

본 프로젝트의 1차 목표는 모델 비교와 데이터 한계 확인이다. 다만 최종 서비스 또는 현업 의사결정 보조 모델로 승격하려면 다음 최소 기준을 만족해야 한다.

운영 후보 통과 기준:

```text
WAPE <= 10%
MAE <= target scale의 5% 수준
filtered MAPE <= 20%
```

권장 목표 기준:

```text
WAPE <= 5%
MAE <= target scale의 2% 수준
filtered MAPE <= 15%
```

판정 규칙:

- WAPE가 10%를 넘는 모델은 운영/서비스 투입 후보에서 제외한다.
- WAPE가 30% 이상이면 모델 구조 문제가 아니라 현재 feature set의 정보 부족 가능성을 우선 의심한다.
- 모든 SOTA 모델을 적용해도 WAPE가 10% 아래로 내려오지 않으면, `core_11`, `chem_22`, `chem_derived` feature만으로는 `remain_capacity` 예측에 필요한 정보가 부족하다고 결론낸다.
- 이 경우 모델을 계속 복잡하게 하기보다 신규 feature 설계, 측정 조건 추가, 시계열 데이터 확보, 또는 문제 정의 변경을 우선 검토한다.
- 위 기준은 target-derived leakage feature를 제외한 정상 feature set에서만 적용한다.

Leakage 기준:

- `remain_capacity = discharge_capacity (mAh/g) * state_of_charge / 100` 관계가 전체 Training/Validation CSV에서 성립한다.
- `discharge_capacity (mAh/g)`와 `state_of_charge`를 함께 feature로 사용한 실험은 target leakage 실험으로 분류한다.
- 위 두 컬럼을 포함하는 `official`/raw-full feature set은 실행 옵션으로 제공하지 않는다.
- 과거에 생성된 해당 조합 결과는 target leakage artifact로만 보관하고, 최종 모델 후보 또는 운영 후보 순위에는 포함하지 않는다.

1차 목표:

- `LightGBM` 또는 `CatBoost`로 재현 가능한 baseline 확보
- RMSE/MAE/WAPE 기준으로 재현 가능한 baseline 확보
- 문서상 MAPE benchmark는 target 0 처리 방식과 target-derived feature 사용 여부를 확인한 뒤 참고용으로만 비교

2차 목표:

- 딥러닝 계열 모델 중 최소 1개 이상이 GBDT baseline과 경쟁 가능한 성능을 보이는지 확인
- Foundation model이 샘플링 데이터에서 GBDT/FT-Transformer 대비 실험 가치가 있는지 확인

### 3.2 실험 품질 기준

- random split과 group split을 모두 수행한다.
- group split 기준은 가능한 경우 `DOI` 또는 `material_id`를 우선 사용한다.
- 메타데이터 칼럼은 학습 입력에서 제외한다.
- target-derived leakage 칼럼인 `discharge_capacity (mAh/g)`, `state_of_charge`는 정상 모델 학습 입력에서 제외한다.
- 1차 모델 개발에서는 `core_11` 11개 feature만 사용한다.
- `design_15`, `chem_22`, `chem_derived`는 `core_11`에서 우수한 모델을 찾은 뒤 후순위 별도 모델로 개발한다.
- `chem_derived`는 `chem_22` 이후의 feature engineering 효과 검증용 feature set이며, 먼저 LightGBM으로 효용을 검증한다.
- 48개 raw column 전체를 정상 leaderboard용 입력으로 자동 사용하지 않는다.
- `official`/raw-full feature set은 코드에서 지원하지 않는다.
- 모든 실험은 seed, split 방식, 샘플 수, 모델 hyperparameter, 실행 시간을 기록한다.
- 결과는 재현 가능한 CSV/JSON 형태로 저장한다.
- sample 기반 실험은 기본적으로 `source_family` 비율을 유지하는 stratified sampling을 사용한다.
- random sampling을 사용하는 경우 NCM 편중 효과를 결과 해석에 명시한다.
- 모든 주요 metric은 전체 validation metric과 `source_family`별 metric을 함께 기록한다.

## 4. 실험 범위

## 4.1 Tier 0: Sanity Check

목적: 데이터 처리, split, metric 계산, leakage 위험을 빠르게 검증한다.

모델:

- Mean predictor
- Median predictor
- Ridge 또는 ElasticNet

필수 확인:

- `remain_capacity` 음수 또는 1000 이상 outlier 필터링 전후 분포
- 결측치 비율
- 범주형 cardinality
- random split과 group split의 성능 차이

## 4.2 Tier 1: Full Data GBDT Baseline

목적: 전체 데이터에서 가장 강한 실무 baseline을 만든다.

모델:

- LightGBM
- CatBoost

권장 설정:

- LightGBM
  - categorical feature 명시
  - objective: regression
  - metric: RMSE, MAE, WAPE, SMAPE, filtered MAPE, raw MAPE 별도 계산
  - early stopping 사용
- CatBoost
  - categorical feature native 처리
  - loss_function: RMSE 또는 MAE
  - depth: 6, 8, 10 sweep
  - learning_rate, l2_leaf_reg, iterations sweep
  - Python 3.14t/free-threaded 공식 wheel이 없으면 `.venv314` 공식 wheel 환경에서 실행
  - `.venv314t`에서 소스 빌드하지 않음

실험 단위:

- 100k sample
- 1M sample
- full data

## 4.3 Tier 2: Large-Subset / Full-Data Neural Baseline

목적: 전체 또는 대규모 subset에서 실용 가능한 정형 딥러닝 모델을 비교한다.

모델:

- RealMLP
- TabM
- TabR
- DCN-V2
- NODE

추천 이유:

- RealMLP: 강한 MLP baseline
- TabM: 최근 tabular benchmark에서 강한 효율적 MLP ensemble 계열
- TabR: retrieval-augmented tabular DL로 유사 recipe 검색 기반 개선 가능성 확인
- DCN-V2: 소재 조합과 전압 조건의 explicit feature interaction 검증에 적합
- NODE: differentiable oblivious decision tree ensemble로 tree-like neural 비교축 제공

실험 단위:

- 100k sample
- 1M sample
- 가능하면 3M 이상
- full data는 리소스가 허용될 때만 수행

DCN-V2 주요 sweep:

- cross layers: 2, 3, 4
- MLP hidden layers: 2, 3
- embedding dimension: 16, 32, 64
- low-rank cross network 사용 가능 시 rank sweep

## 4.4 Tier 3: Transformer / Foundation Model

목적: 최신 정형 데이터 트랜스포머 및 파운데이션 모델의 실험 가치를 검증한다.

모델:

- FT-Transformer
- TabTransformer
- TabPFN-3
- TabPFN latest, 기본 `v3`
- TabICLv2
- TabNet

역할:

- FT-Transformer: 직접 학습형 tabular transformer 기준 모델
- TabTransformer: 범주형 feature attention 특화 비교 모델
- TabPFN-3: 최신 Tabular Foundation Model, 샘플링 데이터 중심
- TabPFN latest: 최신 Prior Labs checkpoint/API 접근 가능 시 `tabpfn_latest`로 별도 기록
- TabICLv2: 대용량 지향 In-Context Learning 기반 foundation model
- TabNet: 해석 가능성과 feature selection 관점의 보조 비교 모델

실험 단위:

- 50k sample
- 100k sample
- 500k sample
- 1M sample
- 그 이상은 하드웨어와 실행 시간에 따라 선택

주의:

- Training 데이터는 NCM이 약 67%를 차지하므로 sample 기반 foundation/transformer 실험은 stratified sampling을 기본으로 한다.
- TabPFN-3는 `core_11`처럼 feature 수가 적은 입력에서는 100k~1M row sample 실험 가치가 있다.
- TabPFN-2.6 fallback은 100k row급까지 우선 검토하고, TabPFN-2.5 fallback은 50k~100k row급으로 제한한다.
- TabPFN-3와 TabICLv2는 공개 로컬/OSS 경로 기준 전체 16.3M건 학습용 기본 모델로 전제하지 않는다.
- TabPFN Enterprise/API Scaling Mode 또는 Large Data Mode를 사용할 경우 일반 로컬 `tabpfn_latest` 결과와 분리해 기록한다.
- 샘플 수를 단계적으로 늘리며 OOM, 실행 시간, 성능을 기록한다.
- Foundation model 결과는 full-data GBDT 결과와 직접 동등 비교하지 않고, 동일 sample 조건에서 비교한다.

## 5. 전처리 요구사항

### 5.1 공통 전처리

- 학습 제외 메타데이터 칼럼 제거
- 타겟 `remain_capacity` 결측 제거
- `remain_capacity < 0` 또는 `remain_capacity >= 1000` 후보 outlier 필터링
- 입력 feature 결측치 처리
  - GBDT: 모델 native missing 처리 우선
  - 딥러닝: 명시적 missing category 또는 imputation 적용
- 범주형 변수는 문자열 또는 category dtype으로 관리
- 수치형 변수는 딥러닝 모델에서 standard scaling 적용

### 5.2 모델별 인코딩

- LightGBM/CatBoost
  - 가능한 경우 native categorical 처리
  - one-hot은 cardinality가 낮은 경우에만 보조 실험
- 딥러닝 모델
  - categorical ID encoding + embedding
  - unknown category index 확보
  - validation/test에만 존재하는 category 처리
- TabPFN/TabICL
  - 각 라이브러리의 권장 preprocessing 우선
  - 필요 시 categorical encoding 방식 명시 기록

### 5.3 파생변수 생성 정책

파생변수는 `chem_22` 이후의 별도 feature set인 `chem_derived`에서만 사용한다.

생성 변수:

```text
voltage_window = voltage_range(V)_max - voltage_range(V)_min
voltage_mid = (voltage_range(V)_max + voltage_range(V)_min) / 2
Ni_to_Mn = Ni_fraction / (Mn_fraction + eps)
Ni_to_Co = Ni_fraction / (Co_fraction + eps)
Li_to_TM = Li_fraction / (Ni_fraction + Mn_fraction + Co_fraction + eps)
active_to_binder = active_proportion / (binder_proportion + eps)
total_transition_metal = Ni_fraction + Mn_fraction + Co_fraction
```

규칙:

- `core_11`은 최소 recipe 입력 모델로 보존한다.
- `chem_22`는 원본 확장 feature set으로 보존한다.
- `chem_derived`는 원본 feature set과 같은 leaderboard 행으로 섞지 않는다.
- `remain_capacity`, `discharge_capacity (mAh/g)`, `state_of_charge`를 사용하는 파생변수는 금지한다.
- `chem_derived`의 1차 검증은 LightGBM으로 수행한다.
- LightGBM에서 `chem_22` 대비 명확한 개선이 있을 때 CatBoost, TabM, NODE, TabPFN/TabICL 후보로 확장한다.

## 6. Split 전략

### 6.1 Random Split

목적:

- 일반적인 benchmark 성능 확인
- 기존 문서상 benchmark와 비교 가능한 기준 확보

기본 비율:

- train: 80%
- validation: 10%
- test: 10%

### 6.2 Group Split

목적:

- 논문/소재 단위 leakage 가능성 검증
- 실제 신규 소재 또는 신규 문헌 예측 상황에 가까운 평가

우선순위:

1. `DOI` 기준 group split
2. `material_id` 기준 group split
3. 둘 다 불가능하면 `chemical_formula` 기준 group split 후보 검토

주의:

- group key는 학습 입력 feature로 사용하지 않는다.
- group split 성능이 random split보다 크게 낮으면 random split 성능을 실무 성능으로 과신하지 않는다.

## 7. 실험 결과 기록 포맷

각 실험은 최소 다음 항목을 기록한다.

```text
experiment_id
model_name
model_family
pretrained
checkpoint
weight_source
access_mode
license_checked
training_mode
data_size
sample_size
split_type
split_seed
group_key
features_used
target_transform
preprocessing_version
hyperparameters
train_time_sec
predict_time_sec
peak_memory_mb
gpu_memory_mb
valid_mape
valid_mae
valid_rmse
valid_wape
valid_smape
valid_filtered_mape
valid_filtered_mape_threshold
valid_filtered_mape_n_rows
test_mape
test_mae
test_rmse
test_wape
test_smape
test_filtered_mape
test_filtered_mape_threshold
test_filtered_mape_n_rows
notes
```

권장 산출 파일:

- `results/experiments.csv`
- `results/leaderboard_random.csv`
- `results/leaderboard_group.csv`
- `results/model_artifacts/`
- `results/plots/`

## 8. 모델 우선순위

### 8.1 전체 우선순위

```text
1. LightGBM
2. CatBoost
3. RealMLP
4. TabM
5. TabR
6. DCN-V2
7. NODE
8. FT-Transformer
9. TabPFN-3/latest
10. TabICLv2
11. TabTransformer
12. TabNet
13. AutoGluon/Mitra, ceiling only
```

### 8.2 Full Data 우선순위

```text
1. LightGBM
2. CatBoost
3. RealMLP
4. TabM
5. TabR, sample/index 비용 허용 시
6. DCN-V2, 리소스 허용 시
7. NODE, 100k/500k에서 유망할 때
```

### 8.3 Sampled Data 우선순위

```text
1. LightGBM
2. CatBoost
3. FT-Transformer
4. DCN-V2
5. TabR
6. NODE
7. TabPFN-3/latest
8. TabICLv2
9. TabM
10. RealMLP
11. TabTransformer
12. TabNet
13. AutoGluon/Mitra, ceiling only
```

## 9. 주요 비교 질문

본 실험은 단순 leaderboard가 아니라 다음 질문에 답해야 한다.

1. GBDT baseline은 RMSE/MAE/WAPE 기준으로 안정적인가?
2. random split과 group split의 성능 차이는 어느 정도인가?
3. CatBoost의 categorical 처리와 LightGBM의 categorical 처리는 어느 쪽이 더 강한가?
4. DCN-V2의 explicit cross 구조는 배터리 소재 조합 예측에서 CatBoost/LightGBM보다 유리한가?
5. RealMLP/TabM은 대규모 정형 회귀에서 FT-Transformer보다 실용적인가?
6. FT-Transformer는 수치형 전압 피처까지 token attention에 포함했을 때 의미 있는 성능 이득을 보이는가?
7. TabTransformer는 범주형 9개 중심 구조에서 FT-Transformer 대비 장점이 있는가?
8. TabR의 retrieval 구조는 `core_11`에서 LightGBM이 놓친 유사 recipe 패턴을 보완하는가?
9. TabPFN-3/latest와 TabICLv2는 같은 sample 조건에서 GBDT baseline을 이길 수 있는가?
10. Foundation model의 성능 이득이 설치/라이선스/실행 비용을 정당화하는가?
11. AutoGluon/Mitra ceiling benchmark가 단일 모델 대비 의미 있는 상한선을 보여주는가?
12. 최종 실무 모델은 단일 모델인가, AutoML/ensemble인가?

## 10. 리스크 및 대응

실행 전 상세 체크리스트는 `docs/RUN_RISK_CHECKLIST.md`를 따른다.

### 10.1 데이터 누수

리스크:

- 같은 논문, 소재, 화학식에서 파생된 row가 train/test에 동시에 존재할 수 있다.
- `discharge_capacity (mAh/g)`와 `state_of_charge`는 `remain_capacity`를 직접 복원하는 target-derived feature이다.
- `official`/raw-full feature set은 위 두 컬럼을 포함하므로 실행 옵션에서 제거한다.

대응:

- random split 외에 group split 필수 수행
- `DOI`, `material_id`, `chemical_formula`를 입력 feature에서 제외
- `discharge_capacity (mAh/g)`, `state_of_charge`를 정상 학습 feature에서 제외
- 기존 `official`/raw-full 실험 결과는 별도 태그를 붙이고 최종 운영 후보에서 제외

### 10.2 대용량 학습 비용

리스크:

- 16.3M건 전체 학습 시 GPU/CPU memory, I/O, 학습 시간이 병목이 될 수 있다.
- Neural/Transformer/Foundation 모델은 GPU memory/OOM으로 실패할 수 있다.
- 노트북 RTX 4060 8GB 환경에서는 1M 이상 neural/transformer 실험을 바로 시작하지 않는다.

대응:

- 50k, 100k, 500k, 1M, full 순서로 확장
- 각 단계에서 시간과 memory 기록
- OOM 발생 시 batch size, embedding dimension, model depth 축소
- 대형 실험은 한 번에 하나의 모델만 실행
- 실행 전 `nvidia-smi`로 GPU 점유 프로세스 확인
- OOM 발생 시 실패 실험으로 기록하고, 축소한 최종 batch/model/sample 조건을 로그에 남김

### 10.3 Foundation Model 라이선스 및 접근성

리스크:

- TabPFN-3/2.6/2.5 계열은 라이선스, token, API, Hugging Face 접근 조건이 있을 수 있다.
- TabPFN Enterprise/API Scaling Mode는 비용, 데이터 업로드 제한, 사내 데이터 반출 가능성, 재현성 조건을 별도 검토해야 한다.
- AutoGluon/Mitra는 optional dependency이며 기본 환경에 설치되어 있지 않을 수 있다.
- AutoGluon/Mitra는 내부 모델 구성이 복합적이므로 단일 모델 성능 원인 분석에는 부적합하다.

대응:

- 설치 전 라이선스 확인
- local OSS 사용 가능 여부와 API 사용 여부를 분리 기록
- 상업적 사용 가능성은 별도 검토
- API/Enterprise Scaling Mode는 별도 ceiling/large-scale foundation 실험으로만 실행
- TabPFN은 `TABPFN_TOKEN`, cached token, 또는 local checkpoint가 없으면 실행하지 않고 실패 실험으로 기록
- AutoGluon/Mitra는 `.venv314`에서 설치 가능성을 확인하고, 50k 또는 100k sample smoke 이후에만 ceiling benchmark로 실행
- AutoGluon/Mitra 실행 시 `time_limit`을 반드시 지정

### 10.4 평가 지표 왜곡

리스크:

- `remain_capacity`가 0에 가까운 경우 raw MAPE가 불안정하다.

대응:

- RMSE, MAE, WAPE를 primary metric으로 사용
- SMAPE와 filtered MAPE를 보조 지표로 사용
- raw MAPE는 diagnostic으로만 기록
- 필요 시 target bin별 error 분석 추가

## 11. 구현 단계

### Phase 1: 데이터/평가 파이프라인

- 데이터 로딩
- 공통 feature/target 정의
- outlier 필터링
- random split 구현
- group split 구현
- metric 계산 함수 구현
- experiment logging 구현

완료 조건:

- Tier 0 모델 결과가 `results/experiments.csv`에 기록됨

### Phase 2: GBDT Baseline

- LightGBM 구현
- CatBoost 구현
- 100k, 1M, full data 실험
- random/group split 비교

완료 조건:

- LightGBM/CatBoost leaderboard 생성

### Phase 3: Neural Baseline

- RealMLP 구현
- TabM 구현
- TabR 구현
- DCN-V2 구현
- NODE 구현
- 100k, 1M 실험

완료 조건:

- GBDT 대비 딥러닝 모델 성능/시간 비교표 생성

### Phase 4: Transformer/Foundation Model

- FT-Transformer 구현
- TabTransformer 구현
- TabPFN-3 설치 및 샘플 실험
- TabPFN latest, 기본 v3 접근 가능성 확인 및 샘플 실험
- TabICLv2 설치 및 샘플 실험
- TabNet 보조 실험
- AutoGluon/Mitra ceiling benchmark 선택 실행

완료 조건:

- 동일 sample 조건에서 foundation model과 GBDT/FT-Transformer 비교표 생성
- TabPFN/TabICL 계열은 training-free 또는 in-context 실험으로 분리 기록

### Phase 5: 최종 분석

- random split vs group split 차이 분석
- 모델별 성능/시간/메모리 Pareto 분석
- feature importance 또는 permutation importance 분석
- XAI/OOD/신뢰도 경고 적용 가능성 분석
- 최종 추천 모델 선정

완료 조건:

- 최종 리포트 작성
- 운영 후보 모델과 연구 후보 모델 분리 제안
- XAI 및 monitoring 후속 구현 우선순위 제안

### Phase 6: 배터리 설계 AI 에이전트 후속 검토

Phase 1~5에서 선정된 최종 예측 모델은 향후 배터리 설계 AI 에이전트의 `capacity_predictor_tool`로 연결할 수 있다.

에이전트의 목적은 단순 회귀 예측이 아니라 다음 workflow를 수행하는 것이다.

```text
자연어 설계 목표
-> 탐색 조건/제약 해석
-> 후보 조합 생성
-> capacity predictor 반복 호출
-> 목표 성능 후보 필터링
-> 비용/공급망/문헌/도메인 규칙 결합
-> 후보 조합 리포트 생성
```

주의:

- 에이전트가 제안하는 조합은 실험적으로 검증된 사실이 아니라 모델 기반 가상 후보로 취급한다.
- 최종 출력에는 "예측 기반 추천", "실험 검증 필요", "학습 데이터 분포 밖 후보 여부"를 명시한다.
- 가격 DB, 논문 검색, 특허 검색, RDKit 기반 화학 정보 도구는 Phase 6 이후 후속 범위로 둔다.

## 12. 최종 산출물

- `docs/PRD.md`
- `docs/AGENT_SYSTEM.md`, Phase 6 진행 시
- `docs/XAI_MONITORING.md`
- 실험 코드
- 전처리 파이프라인
- 모델 학습 스크립트
- `results/experiments.csv`
- random split leaderboard
- group split leaderboard
- 모델별 성능/시간/메모리 비교 리포트
- 최종 모델 추천 리포트

## 13. 비범위

이번 PRD에서는 다음을 필수 범위로 두지 않는다.

- 실제 배터리 화학 domain rule 기반 feature engineering
- 논문 텍스트 또는 `chemical_formula` 자연어/화학식 임베딩 활용
- 모델 serving API 구축
- 자연어 기반 AI 에이전트 UI 또는 agent runtime 구축
- 외부 가격 DB, 공급망 API, 특허/논문 검색 API 연동
- 하이퍼파라미터 대규모 AutoML 최적화
- 원천 데이터 재수집 또는 라벨 품질 검증

단, Phase 5 이후 성능 병목이 명확해지면 `chemical_formula` parsing, 조성비 추출, DOI 기반 논문 텍스트 feature 추가를 후속 과제로 검토할 수 있다.
