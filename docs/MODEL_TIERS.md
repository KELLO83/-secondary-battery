# MODEL_TIERS: 모델 실험 로드맵

## 1. 목적

본 문서는 2차전지 정형 회귀 데이터셋에 적용할 모델군, 실험 규모, 기본 hyperparameter 범위, 중단 조건을 정의한다.

모든 실험은 `docs/PRD.md`의 성공 기준과 `docs/SYSTEM_ARCH.md`의 시스템 구조를 따른다.

## 2. 공통 실험 조건

### 2.0 Pretrained Weight 사용 정책

공식 pretrained checkpoint 또는 foundation model weight가 공개되어 있는 모델은 기본적으로 pretrained weight를 사용한다.

적용 대상:

- TabPFN-3
- TabPFN-2.5, fallback 필요 시
- TabICLv2
- TabICL, fallback 필요 시
- 기타 공식 pretrained checkpoint가 제공되는 tabular foundation model

원칙:

- pretrained model 결과와 from-scratch 학습 결과는 분리해서 기록한다.
- checkpoint 이름, 버전, 다운로드 출처, 접근 방식, 라이선스 확인 여부를 반드시 기록한다.
- API 기반 pretrained model과 local checkpoint 기반 pretrained model은 구분한다.
- fine-tuning을 수행하는 경우 zero/few-shot 또는 in-context 결과와 별도 실험으로 기록한다.
- pretrained weight를 사용할 수 없는 경우 임의로 from-scratch 결과로 대체하지 않고, 사유를 `notes`에 기록한다.
- TabPFN 계열은 Prior Labs 라이선스 승인과 token 또는 local checkpoint가 필요할 수 있다. 정식 실험은 브라우저 로그인 프롬프트에 의존하지 않고 `TABPFN_TOKEN`, cached token, 또는 local `model_path`를 사용한다.

구현 정책:

- 외부 pip/오픈소스 구현이 있는 모델은 프로젝트 내부에 모델 본체를 재구현하지 않는다.
- 모델별 wrapper 파일은 모델별로 분리한다.
  - `ml/src/models/neural/RealMLP.py`
  - `ml/src/models/neural/TabM.py`
  - `ml/src/models/neural/DCNV2.py`
  - `ml/src/models/transformer/FTTransformer.py`
  - `ml/src/models/transformer/TabTransformer.py`
  - `ml/src/models/transformer/TabNet.py`
  - `ml/src/models/foundation/TabPFN.py`
  - `ml/src/models/foundation/TabICLv2.py`
- `TabPFN.py`와 `TabICLv2.py`는 공식 pretrained regressor를 import해서 사용하며 from-scratch 학습하지 않는다.
- 선택 dependency는 `ml/requirements-optional-dl.txt`에 둔다.
- `ml/requirements-optional-dl.txt`는 `.venv314` 기준으로 설치한다.
- LightGBM만 `.venv314t`에서 실행하고, LightGBM을 제외한 모든 모델은 `.venv314`에서 실행한다.
- Neural/Transformer/Foundation 모델은 기본적으로 GPU 학습 또는 GPU 추론을 사용하므로 `.venv314`에서 실행한다.
- `.venv314t`에서 PyTorch/CUDA/foundation model wheel이 없으면 소스 빌드하지 않고 `.venv314` 또는 공식 지원 환경을 사용한다.

### 2.0.1 Training-Free / Inference-Only 모델 정책

모든 모델이 프로젝트 데이터로 gradient 기반 학습을 수행해야 하는 것은 아니다.

Tabular foundation model 계열은 공식 pretrained weight를 사용해 별도 하이퍼파라미터 튜닝 또는 from-scratch 학습 없이 바로 예측 실험을 수행할 수 있다.

적용 대상:

- TabICLv2
- TabICL
- TabPFN-3
- TabPFN-2.5, fallback 필요 시

원칙:

- training-free 모델은 `fit()`이 있더라도 일반적인 학습이 아니라 context/index/cache 준비 단계로 간주한다.
- 실험 로그에는 `training_mode`를 기록한다.
- 가능한 값:

```text
from_scratch
fine_tune
pretrained_inference
in_context
zero_shot
```

- TabICLv2와 TabPFN 계열의 기본값은 `pretrained_inference` 또는 `in_context`로 둔다.
- 이 모델들은 LightGBM/CatBoost처럼 hyperparameter tuning 대상으로 취급하지 않는다.
- 동일 sample/split 조건에서 GBDT baseline과 비교하되, 학습 시간과 추론 시간을 분리 기록한다.

### 2.1 공통 Metric

모든 모델은 다음 지표를 기록한다.

- MAPE
- MAE
- RMSE
- 학습 시간
- 추론 시간
- peak memory
- GPU memory, 해당 시

### 2.2 공통 Split

모든 주요 모델은 가능한 경우 두 가지 split에서 평가한다.

- random split
- group split

group split 우선순위:

1. `DOI`
2. `material_id`
3. `chemical_formula`

### 2.3 공통 Sample Size

기본 실험 규모:

```text
50k
100k
500k
1M
3M
full
```

모델별로 허용되는 최대 규모는 다르며, 아래 tier 정의를 따른다.

### 2.4 통합 모델 학습 정책

본 프로젝트의 기본 학습 단위는 LFP, NCA, NCM, Others를 모두 병합한 통합 데이터셋이다.

원칙:

- 계열별 CSV를 모두 로딩해 하나의 train/validation dataframe으로 병합한다.
- 파일명에서 `source_family`를 파생하되 학습 feature로 사용하지 않는다.
- 모든 모델은 기본적으로 통합 모델 1개만 학습한다.
- validation 결과는 전체 metric과 `source_family`별 metric을 함께 기록한다.
- LFP/NCA/NCM/Others별 specialized model은 초기 실험 범위에서 제외한다.

Feature set 기본값:

```text
phase 1/default: core_11
agent/design-oriented: core_11
phase 2 candidates: design_15, chem_22
ablation/benchmark: official
```

원칙:

- 1차 모델 개발은 `core_11`만 사용한다.
- `core_11`은 연구원이 직접 입력하기 쉬운 11개 핵심 feature만 사용한다.
- 먼저 `core_11`에서 우수한 모델군과 기본 hyperparameter 범위를 찾는다.
- `design_15`, `chem_22`, `official`은 `core_11` 결과가 나온 뒤 후순위 별도 모델로 비교한다.
- 1차 실험에서 48개 CSV 칼럼 전체 또는 `official` feature set을 기본으로 사용하지 않는다.

후순위 feature 확장:

| feature set | 추가 입력 | 목적 |
| :--- | :--- | :--- |
| `design_15` | `sintering_T1(C)`, `sintering_t1(h)`, `measurement_T(C)`, `C-rate` | 합성/측정 조건까지 입력받는 확장 모델 |
| `chem_22` | `Li_fraction`, `Ni_fraction`, `Mn_fraction`, `Co_fraction`, `dopant_fraction`, `active_proportion`, `binder_proportion` | 조성비와 전극 구성 비율까지 입력받는 확장 모델 |

`design_15`와 `chem_22`는 `core_11` 모델과 입력 스키마가 다르므로 같은 실험 ID 또는 같은 leaderboard 행으로 섞지 않는다.

후속 검토 조건:

- 특정 `source_family`에서 지속적으로 높은 error가 발생함
- 해당 계열의 데이터 수가 충분함
- 통합 모델 개선으로 해결되지 않음
- 계열별 모델 추가로 인한 운영 복잡도를 감수할 가치가 있음

## 3. Tier 0: Sanity Check

목적:

- 데이터 로딩, split, metric 계산, logging 검증
- target 분포와 leakage 위험 확인

모델:

- Mean predictor
- Median predictor
- Ridge
- ElasticNet, 선택

Sample size:

- 50k
- 100k
- full, 가능하면 metric만 빠르게 계산

성공 조건:

- 실험 결과가 `results/experiments.csv`에 기록됨
- random split과 group split이 모두 생성됨
- group overlap 검사 통과

중단 조건:

- 필수 칼럼 누락
- target 결측 또는 이상치 처리 규칙 미정
- split 간 group overlap 발생

## 4. Tier 1: Full-Data GBDT Baseline

목적:

- 전체 데이터 기준 실무 baseline 확보
- 이후 딥러닝/foundation model 비교의 기준점 설정

모델:

- LightGBM
- CatBoost

### 4.1 LightGBM

역할:

- 대용량 정형 회귀 baseline
- 빠른 학습과 strong baseline 확보

Sample size:

- 100k
- 1M
- 3M
- full

기본 hyperparameter 후보:

```yaml
objective: regression
learning_rate: [0.03, 0.05, 0.1]
num_leaves: [63, 127, 255]
max_depth: [-1, 8, 12]
min_data_in_leaf: [50, 200, 1000]
feature_fraction: [0.8, 1.0]
bagging_fraction: [0.8, 1.0]
bagging_freq: [0, 1]
n_estimators: 5000
early_stopping_rounds: 100
```

주의:

- categorical feature를 명시한다.
- one-hot encoding은 보조 실험으로만 사용한다.
- MAPE는 별도 함수로 계산한다.

### 4.2 CatBoost

역할:

- 범주형 native 처리 baseline
- 고차원 조건부 feature interaction 비교 기준
- Python 3.14t/free-threaded wheel이 공식 제공되지 않으면 `.venv314`에서 공식 wheel로 실행하는 GBDT baseline

Sample size:

- 100k
- 1M
- 3M
- full

기본 hyperparameter 후보:

```yaml
loss_function: RMSE
eval_metric: RMSE
depth: [6, 8, 10]
learning_rate: [0.03, 0.05, 0.1]
l2_leaf_reg: [3, 10, 30]
iterations: 5000
random_strength: [1, 5, 10]
early_stopping_rounds: 100
```

주의:

- categorical feature index를 명확히 지정한다.
- `depth=10`은 학습시간과 과적합을 반드시 확인한다.
- DCN-V2와 비교할 때는 동일 sample/split 조건에서 비교한다.
- `.venv314t`에서 CatBoost를 소스 빌드하지 않는다.
- `PYTHON_GIL=1`은 CatBoost wheel ABI 문제를 해결하지 못하므로 설치 우회책으로 사용하지 않는다.
- CatBoost 메인 baseline은 `.venv314`에서 GPU를 기본값으로 실행한다.
- CatBoost GPU 실행은 `task_type=GPU`, `devices=0`, `gpu_ram_part=0.90`을 기본값으로 사용해 프로젝트 GPU 90% 정책을 따른다.
- CatBoost CPU 실행은 smoke/debug/fallback 용도로만 사용하며, 필요한 경우 `--task-type CPU`를 명시한다.
- CatBoost CPU fallback은 `.venv314`에서 `thread_count=14`로 실행하고, `python_executable`과 `python_version`을 실험 로그에 기록한다.
- LightGBM은 `.venv314t`, CatBoost는 `.venv314`에서 실행될 수 있으므로 실행 환경 차이를 결과 해석에 명시한다.

## 5. Tier 2: Large-Subset / Full-Data Neural Baseline

목적:

- 트랜스포머보다 가벼운 정형 딥러닝 모델의 실용성을 검증한다.
- GBDT와 비교 가능한 대규모 neural baseline을 확보한다.

모델:

- RealMLP
- TabM
- DCN-V2

### 5.1 RealMLP

역할:

- 강한 MLP baseline
- 트랜스포머가 꼭 필요한지 판단하는 기준

Sample size:

- 100k
- 500k
- 1M
- 3M, 가능하면
- full, 리소스 허용 시

기본 sweep:

```yaml
hidden_dim: [256, 512, 1024]
num_layers: [2, 3, 4]
dropout: [0.0, 0.1, 0.2]
weight_decay: [0.0, 1e-5, 1e-4]
batch_size: [2048, 4096, 8192]
learning_rate: [1e-4, 3e-4, 1e-3]
epochs: [20, 50, 100]
early_stopping_patience: 5
```

### 5.2 TabM

역할:

- 최근 tabular benchmark에서 강한 효율적 MLP ensemble 계열 후보
- full 또는 large-subset neural model 후보

Sample size:

- 100k
- 500k
- 1M
- 3M, 가능하면
- full, 리소스 허용 시

기본 sweep:

```yaml
embedding_dim: [16, 32, 64]
hidden_dim: [256, 512]
num_layers: [2, 3]
dropout: [0.0, 0.1, 0.2]
batch_size: [2048, 4096, 8192]
learning_rate: [1e-4, 3e-4, 1e-3]
weight_decay: [1e-6, 1e-5, 1e-4]
epochs: [20, 50, 100]
```

### 5.3 DCN-V2

역할:

- 소재 조합과 전압 조건의 explicit feature interaction 검증
- CatBoost의 조건부 분할과 neural cross 구조를 비교하는 핵심 모델

Sample size:

- 100k
- 500k
- 1M
- 3M, 가능하면
- full, 리소스 허용 시

기본 sweep:

```yaml
embedding_dim: [16, 32, 64]
cross_layers: [2, 3, 4]
cross_type: ["vector", "matrix", "low_rank"]
low_rank: [16, 32, 64]
mlp_hidden_dim: [256, 512, 1024]
mlp_layers: [2, 3]
dropout: [0.0, 0.1, 0.2]
batch_size: [2048, 4096, 8192]
learning_rate: [1e-4, 3e-4, 1e-3]
weight_decay: [1e-6, 1e-5, 1e-4]
epochs: [20, 50, 100]
early_stopping_patience: 5
```

핵심 비교:

- DCN-V2 vs CatBoost depth 6/8/10
- 동일 sample size
- 동일 split
- 동일 feature set

## 6. Tier 3: Transformer / Foundation Model

목적:

- 최신 정형 데이터 transformer와 foundation model의 실험 가치를 검증한다.
- full-data production 후보라기보다 연구/비교 후보로 평가한다.

모델:

- FT-Transformer
- TabTransformer
- TabPFN-3
- TabICLv2
- TabNet

### 6.1 FT-Transformer

역할:

- 직접 학습형 tabular transformer 기준 모델
- 범주형과 수치형을 모두 token으로 처리하는 구조 비교

Sample size:

- 50k
- 100k
- 500k
- 1M, 리소스 허용 시

기본 sweep:

```yaml
embedding_dim: [32, 64, 128]
num_heads: [4, 8]
num_layers: [2, 4, 6]
ffn_hidden_multiplier: [2, 4]
dropout: [0.0, 0.1, 0.2]
batch_size: [1024, 2048, 4096]
learning_rate: [1e-4, 3e-4, 1e-3]
weight_decay: [1e-5, 1e-4]
epochs: [20, 50, 100]
```

### 6.2 TabTransformer

역할:

- 범주형 9개 feature 간 contextual embedding 비교
- 수치형 전압 변수를 attention 밖에서 결합하는 구조의 장단점 확인

Sample size:

- 50k
- 100k
- 500k
- 1M, 리소스 허용 시

기본 sweep:

```yaml
embedding_dim: [32, 64, 128]
num_heads: [4, 8]
num_layers: [2, 4, 6]
mlp_hidden_dim: [256, 512]
dropout: [0.0, 0.1, 0.2]
batch_size: [1024, 2048, 4096]
learning_rate: [1e-4, 3e-4, 1e-3]
epochs: [20, 50, 100]
```

### 6.3 TabPFN-3

역할:

- 최신 Tabular Foundation Model 후보
- zero/few-shot 또는 in-context 기반 성능 확인
- 공식 pretrained checkpoint 또는 API 모델 사용을 기본값으로 둔다.
- 기본 training mode는 `pretrained_inference` 또는 `in_context`이다.
- Prior Labs 라이선스 승인/token 또는 local checkpoint가 없으면 실험을 실패 처리하고, 임의 from-scratch 대체를 금지한다.

Sample size:

- 50k
- 100k
- 500k
- 1M, 가능하면

주의:

- 라이선스와 접근 조건을 먼저 확인한다.
- local OSS, API, Hugging Face checkpoint 사용 여부를 기록한다.
- full-data GBDT와 직접 비교하지 않고 동일 sample 조건에서 비교한다.
- 사용 버전, checkpoint, API endpoint를 결과에 기록한다.

기록 항목:

```yaml
tabpfn_version: null
checkpoint: null
pretrained: true
training_mode: "pretrained_inference"
access_mode: ["local", "api", "huggingface"]
weight_source: null
license_checked: true
```

### 6.4 TabICLv2

역할:

- 대용량 지향 In-Context Learning 기반 foundation model
- TabPFN-3와 함께 최신 foundation model 비교 축
- 공식 pretrained checkpoint 사용을 기본값으로 둔다.
- 기본 training mode는 `pretrained_inference` 또는 `in_context`이다.
- 프로젝트 데이터로 from-scratch 학습하지 않는다.

Sample size:

- 50k
- 100k
- 500k
- 1M
- 그 이상은 리소스 확인 후 선택

주의:

- CPU/disk offloading 설정 여부를 기록한다.
- checkpoint version을 기록한다.
- OOM 또는 과도한 실행 시간이 발생하면 sample size를 낮춘다.

기록 항목:

```yaml
tabicl_version: null
checkpoint_version: null
pretrained: true
training_mode: "in_context"
weight_source: null
offloading: null
device: null
```

### 6.5 TabNet

역할:

- 해석 가능성과 feature selection 관점의 보조 비교 모델
- 최고 성능 후보보다는 분석용 모델

Sample size:

- 50k
- 100k
- 500k

기본 sweep:

```yaml
n_d: [16, 32, 64]
n_a: [16, 32, 64]
n_steps: [3, 5]
gamma: [1.3, 1.5, 2.0]
lambda_sparse: [1e-5, 1e-4, 1e-3]
batch_size: [1024, 2048, 4096]
learning_rate: [1e-3, 3e-3]
epochs: [50, 100]
```

## 7. AutoGluon 선택 실험

역할:

- 단일 모델 실험 이후 score ceiling을 확인하는 AutoML/ensemble 후보

실행 시점:

- Tier 1, Tier 2, Tier 3 단일 모델 leaderboard가 만들어진 이후

주의:

- AutoGluon 결과는 단일 모델 결과와 분리하여 기록한다.
- 내부에 GBDT, neural, foundation model이 섞일 수 있으므로 모델별 원인 분석에는 부적합하다.
- time limit을 반드시 기록한다.

Sample size:

- 100k
- 500k
- 1M
- full, time budget 허용 시

## 8. 앙상블 정책

초기 실험에서는 voting, stacking, weighted ensemble을 수행하지 않는다.

앙상블은 다음 조건을 만족한 뒤 Phase 5에서 선택적으로 수행한다.

- random split leaderboard 완성
- group split leaderboard 완성
- 단일 모델별 test metric 확보
- inference cost 비교 완료
- leakage 검증 완료

앙상블 후보:

- LightGBM + CatBoost
- GBDT + DCN-V2
- GBDT + TabM
- AutoGluon ensemble

## 9. 우선 실행 순서

```text
1. Tier 0 sanity check
2. LightGBM 100k random/group
3. CatBoost 100k random/group
4. LightGBM 1M
5. CatBoost 1M
6. RealMLP 100k/500k
7. DCN-V2 100k/500k
8. TabM 100k/500k
9. FT-Transformer 50k/100k
10. TabPFN-3 50k/100k
11. TabICLv2 50k/100k
12. 상위 모델만 500k/1M 확장
13. GBDT full-data 실행
14. 최종 Pareto 분석
```

## 10. 모델별 중단 조건

공통 중단 조건:

- 같은 sample/split에서 LightGBM 대비 MAPE가 20% 이상 나쁘고 개선 가능성이 낮음
- 학습 시간이 같은 sample의 GBDT 대비 10배 이상이며 성능 이득이 없음
- OOM이 반복 발생하고 batch/model 축소 후에도 해결되지 않음
- 라이선스 또는 접근 조건이 실험 목적과 맞지 않음

Foundation model 중단 조건:

- 100k 이하 sample에서도 실행 시간이 과도함
- 동일 sample의 CatBoost/LightGBM보다 명확히 낮은 성능
- 라이선스 또는 API 의존성 때문에 재현성이 낮음

## 11. 최종 추천 기준

최종 모델은 단순 성능만으로 선정하지 않는다.

우선순위:

1. group split test MAPE
2. group split test MAE/RMSE
3. random split과 group split의 성능 차이
4. 학습 시간
5. 추론 시간
6. 메모리 사용량
7. 재현성
8. 라이선스/배포 가능성
9. 해석 가능성

운영 후보와 연구 후보는 분리한다.

- 운영 후보: LightGBM, CatBoost, RealMLP, TabM, DCN-V2
- 연구 후보: FT-Transformer, TabTransformer, TabPFN-3, TabICLv2, TabNet

## 12. AI 에이전트 연계 후보 기준

최종 예측 모델은 향후 `docs/AGENT_SYSTEM.md`의 배터리 설계 에이전트에서 `capacity_predictor_tool`로 사용될 수 있다.

에이전트 연계 우선 후보는 다음 조건을 만족해야 한다.

- group split 성능이 random split 대비 과도하게 붕괴하지 않음
- 추론 시간이 대량 후보 screening에 적합함
- 모델 artifact 저장과 재로딩이 안정적임
- 입력 feature schema가 명확함
- 예측 불확실성 또는 out-of-distribution 경고를 붙일 수 있음

에이전트 tool 후보 우선순위:

```text
1. CatBoost 또는 LightGBM best model
2. DCN-V2, 성능과 추론 비용이 충분히 좋은 경우
3. TabM 또는 RealMLP
4. Foundation model은 연구용 tool로 제한
```
