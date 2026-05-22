# SYSTEM_ARCH: 2차전지 Tabular 실험 시스템 아키텍처

## 1. 목적

본 문서는 `docs/PRD.md`와 `docs/DATASETEXPLAIN.md`에 정의된 실험을 구현하기 위한 코드 구조, 모듈 책임, 공통 인터페이스, 결과 저장 규칙을 고정한다.

코드 작성 시 본 문서를 우선 참조하여 한 파일에 모든 로직이 몰리는 것을 방지하고, 모델이 추가되어도 동일한 실험 runner와 평가 파이프라인을 재사용할 수 있게 한다.

## 2. 설계 원칙

- 데이터 스키마는 `docs/DATASETEXPLAIN.md`를 단일 기준으로 삼는다.
- 실험 목표와 성공 기준은 `docs/PRD.md`를 따른다.
- 모델별 세부 실험 범위는 `docs/MODEL_TIERS.md`를 따른다.
- 모든 모델은 가능한 한 동일한 `BaseModel` 인터페이스를 따른다.
- 전처리, split, 학습, 평가, 결과 기록은 분리한다.
- 실험 결과는 사람이 읽을 수 있는 CSV와 재현 가능한 JSON config로 남긴다.
- 모델 추가는 새 파일 1개와 registry 등록만으로 가능해야 한다.

## 3. 권장 디렉터리 구조

```text
secanday_battery/
  AGENT.md
  docs/
    DATASETEXPLAIN.md
    PRD.md
    SYSTEM_ARCH.md
    MODEL_TIERS.md
    AGENT_SYSTEM.md
    SERVICE_ARCH.md
    XAI_MONITORING.md
    DATA_CONTRACT.md
    EVALUATION_PROTOCOL.md
    TABULAR_DATA_VARIABLES.md
  ml/
    requirements-314.txt
    requirements-314t.txt
    requirements-optional-dl.txt
    configs/
      base.yaml
      data.yaml
      models/
        lightgbm.yaml
        catboost.yaml
        realmlp.yaml
        tabm.yaml
        dcnv2.yaml
        ft_transformer.yaml
        tab_transformer.yaml
        tabpfn.yaml
        tabicl.yaml
        tabnet.yaml
    src/
      __init__.py
      schema.py
      data/
        __init__.py
        loader.py
        preprocessing.py
        splitters.py
        sampling.py
      models/
        __init__.py
        base.py
        registry.py
        gbdt/
          __init__.py
          lightgbm_model.py
          catboost_model.py
        neural/
          __init__.py
          RealMLP.py
          TabM.py
          DCNV2.py
        transformer/
          __init__.py
          FTTransformer.py
          TabTransformer.py
          TabNet.py
        foundation/
          __init__.py
          TabPFN.py
          TabICLv2.py
      eval/
        __init__.py
        metrics.py
        evaluator.py
        leakage.py
      experiments/
        __init__.py
        runner.py
        logger.py
        leaderboard.py
      utils/
        __init__.py
        seed.py
        timing.py
        memory.py
        io.py
    scripts/
      run_model.py
      run_experiment.py
      run_tier0.py
      run_gbdt.py
      run_neural.py
      run_foundation.py
      build_leaderboard.py
    tests/
      test_schema.py
      test_metrics.py
      test_splitters.py
      test_preprocessing.py
      test_experiment_logging.py
  backend/
    app/
  frontend/
    src/
  results/
    experiments.csv
    leaderboard_random.csv
    leaderboard_group.csv
    plots/
    model_artifacts/
```

## 3.1 코드 경계

ML 학습 코드는 `ml/` 아래에만 둔다.

서비스/API 코드는 `backend/` 아래에만 둔다.

프론트엔드 코드는 `frontend/` 아래에만 둔다.

원칙:

- ML 실행 환경은 단순하게 구분한다.
- LightGBM CPU baseline만 `.venv314t`에서 실행한다.
- LightGBM을 제외한 모든 모델은 `.venv314`에서 실행한다.
- CatBoost baseline은 `.venv314`에서 실행한다.
- Neural/Transformer/Foundation 모델은 GPU 학습 또는 GPU 추론을 기본으로 하므로 `.venv314`에서 실행한다.
- `backend/`는 `.venv314` 환경을 기본으로 사용한다.
- `backend/`는 모델을 재학습하지 않고 `results/model_artifacts/`의 저장된 artifact를 로드한다.
- `frontend/`는 Python 가상환경을 사용하지 않고 Node.js 패키지 환경을 사용한다.

GBDT 실행 환경:

- LightGBM: `.venv314t` 기본 실행
- CatBoost: `.venv314` 기본 실행, 공식 PyPI wheel 사용, 메인 baseline은 `task_type=GPU`, `devices=0`, `gpu_ram_part=0.90` 기본값
- Neural/Transformer/Foundation: `.venv314` 기본 실행, PyTorch/CUDA 및 pretrained checkpoint 호환성 우선
- CatBoost는 `.venv314t`에서 소스 빌드하지 않는다.
- PyTorch/CUDA/foundation model package도 `.venv314t`에서 소스 빌드하지 않는다.
- 실험 runner는 `python_executable`, `python_version`을 결과 로그에 기록한다.

## 4. 핵심 모듈 책임

### 4.1 `ml/src/schema.py`

역할:

- 데이터 칼럼명을 중앙에서 관리한다.
- 입력 feature, target, metadata column을 상수로 정의한다.
- 칼럼명 오타를 방지한다.

필수 상수:

```python
TARGET_COLUMN = "remain_capacity"

CORE_11_NUMERIC_COLUMNS = [
    "voltage_range(V)_min",
    "voltage_range(V)_max",
]

CORE_11_CATEGORICAL_COLUMNS = [
    "material_structure",
    "synthesis_method",
    "Li_source",
    "Ni_source",
    "Co_source",
    "Mn_source",
    "electrolyte",
    "separator",
    "counter_electrode",
]

METADATA_COLUMNS = [
    "material_id",
    "chemical_formula",
    "DOI",
    "journal_name",
    "Class",
    "Unnamed: 0",
]

DESIGN_15_NUMERIC_COLUMNS = CORE_11_NUMERIC_COLUMNS + [
    "sintering_T1(C)",
    "sintering_t1(h)",
    "measurement_T(C)",
    "C-rate",
]

CHEM_22_NUMERIC_COLUMNS = DESIGN_15_NUMERIC_COLUMNS + [
    "Li_fraction",
    "Ni_fraction",
    "Mn_fraction",
    "Co_fraction",
    "dopant_fraction",
    "active_proportion",
    "binder_proportion",
]
```

규칙:

- 기본 feature set은 `core_11`이다.
- `design_15`, `chem_22`는 `core_11`에서 우수한 모델을 찾은 뒤 후순위 별도 모델로 학습한다.
- 48개 raw column 전체를 자동 feature로 사용하는 helper를 만들지 않는다.

### 4.2 `ml/src/data/loader.py`

역할:

- CSV 또는 parquet 데이터를 로딩한다.
- 필요한 칼럼 존재 여부를 검증한다.
- 대용량 데이터를 위해 필요 시 chunk 또는 lazy loading을 지원한다.

필수 기능:

- `load_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame`
- `validate_columns(df: pd.DataFrame) -> None`

### 4.3 `ml/src/data/preprocessing.py`

역할:

- 공통 전처리를 수행한다.
- 모델 family별 전처리 pipeline을 구성한다.

필수 기능:

- metadata column 제거
- target 결측 제거
- `remain_capacity < 0` 또는 `remain_capacity >= 1000` 필터링
- categorical dtype 변환
- 딥러닝용 categorical id encoding
- 딥러닝용 numerical standard scaling
- validation/test unknown category 처리

### 4.4 `ml/src/data/splitters.py`

역할:

- random split과 group split을 구현한다.

필수 기능:

- `make_random_split(df, seed, train_size, valid_size, test_size)`
- `make_group_split(df, group_col, seed, train_size, valid_size, test_size)`

주의:

- group split의 group key는 학습 feature에 포함하지 않는다.
- split 결과는 row index 또는 boolean mask로 저장 가능해야 한다.

### 4.5 `ml/src/data/sampling.py`

역할:

- 샘플링 실험을 재현 가능하게 만든다.

필수 sample size:

- 50k
- 100k
- 500k
- 1M
- 3M
- full

필수 기능:

- stratified sampling 후보: target bin 기반
- group-aware sampling 후보: DOI/material_id 기준
- seed 고정

### 4.5.1 `source_family` 처리

CSV 파일명에서 `source_family`를 파생한다.

예시:

```text
LFP_train_dataset.csv -> LFP
NCA_train_dataset.csv -> NCA
NCM_train_dataset.csv -> NCM
others_train_dataset.csv -> Others
```

규칙:

- `source_family`는 metadata column으로 관리한다.
- 모델 학습 feature에는 포함하지 않는다.
- 계열별 metric, OOD, drift monitoring, 리포트 필터링에만 사용한다.
- 기본 학습은 4개 계열을 병합한 통합 모델로 수행한다.

### 4.6 `ml/src/models/base.py`

역할:

- 모든 모델 wrapper의 공통 인터페이스를 정의한다.

권장 인터페이스:

```python
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np


class BaseModel(ABC):
    name: str
    family: str

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
```

### 4.7 `ml/src/models/registry.py`

역할:

- 문자열 model name을 실제 model class에 매핑한다.
- runner가 모델별 if/else로 지저분해지는 것을 방지한다.

예시:

```python
MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "realmlp": RealMLPModel,
    "tabm": TabMModel,
    "dcnv2": DCNV2Model,
    "ft_transformer": FTTransformerModel,
    "tab_transformer": TabTransformerModel,
    "tabpfn": TabPFNModel,
    "tabicl": TabICLModel,
    "tabnet": TabNetModel,
}
```

### 4.8 `ml/src/eval/metrics.py`

역할:

- 모든 모델에 같은 metric을 적용한다.

필수 metric:

- MAPE
- MAE
- RMSE

주의:

- MAPE는 target이 0에 가까운 경우 불안정하므로 epsilon을 둔다.
- metric 함수는 numpy array를 입력받고 float를 반환한다.

### 4.9 `ml/src/eval/leakage.py`

역할:

- train/valid/test 간 group overlap을 점검한다.

필수 기능:

- DOI overlap 확인
- material_id overlap 확인
- chemical_formula overlap 확인
- group split 검증 리포트 생성

### 4.10 `ml/src/experiments/runner.py`

역할:

- config를 받아 하나의 실험을 실행한다.

실행 순서:

1. config 로드
2. seed 고정
3. 데이터 로딩
4. 샘플링
5. split 생성
6. 전처리 fit/transform
7. 모델 생성
8. 학습
9. valid/test 예측
10. metric 계산
11. 결과 기록
12. artifact 저장

### 4.11 `ml/src/experiments/logger.py`

역할:

- 실험 결과를 `results/experiments.csv`에 누적 기록한다.
- hyperparameter와 config snapshot을 JSON으로 저장한다.

필수 기록 항목은 `docs/PRD.md`의 "실험 결과 기록 포맷"을 따른다.

## 5. Config 설계

모든 실험은 config-driven으로 실행한다.

예시:

```yaml
experiment:
  id: "lgbm_100k_random_seed42"
  seed: 42

data:
  path: "data/train.csv"
  sample_size: 100000
  target: "remain_capacity"
  outlier:
    min_target: 0
    max_target: 1000

split:
  type: "random"
  train_size: 0.8
  valid_size: 0.1
  test_size: 0.1
  group_key: null

model:
  name: "lightgbm"
  family: "gbdt"
  training_mode: "from_scratch"
  pretrained:
    enabled: false
    checkpoint: null
    weight_source: null
    access_mode: null
    license_checked: null
  params:
    objective: "regression"
    learning_rate: 0.05
    num_leaves: 255
    n_estimators: 5000

output:
  results_dir: "results"
  save_model: true
```

pretrained weight가 있는 모델은 `model.pretrained.enabled: true`로 기록한다.

예시:

```yaml
model:
  name: "tabicl"
  family: "foundation"
  training_mode: "in_context"
  pretrained:
    enabled: true
    checkpoint: "<official-tabicl-v2-regressor-checkpoint>"
    weight_source: "official"
    access_mode: "local"
    license_checked: true
  params:
    device: "cuda"
    offloading: "cpu_disk"
```

주의:

- checkpoint 이름은 실제 task에 맞는 regressor/classifier 여부를 확인한 뒤 기록한다.
- pretrained 결과와 from-scratch 결과는 같은 leaderboard에 표시하되 `pretrained` 컬럼으로 구분한다.
- `training_mode` 컬럼으로 from-scratch, fine-tune, pretrained inference, in-context, zero-shot 실험을 구분한다.
- API 기반 모델은 checkpoint 대신 API model version 또는 endpoint version을 기록한다.

## 6. 결과 저장 규칙

### 6.1 공통 결과

모든 실험은 다음 파일 중 최소 하나에 기록되어야 한다.

- `results/experiments.csv`
- `results/experiments.jsonl`

### 6.2 모델 artifact

모델 artifact는 다음 규칙으로 저장한다.

```text
results/model_artifacts/{experiment_id}/
  model.*
  config.yaml
  metrics.json
  feature_info.json
```

### 6.3 Leaderboard

leaderboard는 split type별로 분리한다.

- `results/leaderboard_random.csv`
- `results/leaderboard_group.csv`

정렬 기준:

1. `test_mape` ascending
2. `test_mae` ascending
3. `test_rmse` ascending
4. `train_time_sec` ascending

## 7. 테스트 요구사항

필수 테스트:

- `test_schema.py`: feature/target/metadata 칼럼 정의 검증
- `test_metrics.py`: MAPE, MAE, RMSE 계산 검증
- `test_splitters.py`: random/group split 비율과 group overlap 검증
- `test_preprocessing.py`: metadata 제거, outlier 필터링, unknown category 처리 검증
- `test_experiment_logging.py`: 결과 CSV/JSON 기록 검증

## 8. 구현 순서

1. `ml/src/schema.py`
2. `ml/src/eval/metrics.py`
3. `ml/src/data/loader.py`
4. `ml/src/data/splitters.py`
5. `ml/src/data/preprocessing.py`
6. `ml/src/experiments/logger.py`
7. `ml/src/models/base.py`
8. `ml/src/models/registry.py`
9. Tier 0 모델
10. LightGBM/CatBoost
11. Neural/foundation 모델

## 9. 금지사항

- 모델별 train script마다 metric 계산을 따로 구현하지 않는다.
- 칼럼명을 코드 여러 곳에 하드코딩하지 않는다.
- 메타데이터 칼럼을 학습 feature로 사용하지 않는다.
- random split 결과만 보고 최종 모델을 결정하지 않는다.
- foundation model 결과를 full-data GBDT 결과와 다른 sample 조건에서 직접 비교하지 않는다.

## 10. AI 에이전트 시스템 연계 구조

Phase 5 이후 선정된 best model은 배터리 설계 AI 에이전트에서 다음 tool로 노출할 수 있다.

```text
capacity_predictor_tool(input: BatteryRecipe) -> CapacityPrediction
```

권장 tool input:

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

권장 tool output:

```text
predicted_remain_capacity
model_name
model_version
split_validation_summary
input_schema_version
is_out_of_distribution
warning
```

주의:

- agent는 모델을 재학습하지 않고 저장된 best model artifact를 호출한다.
- agent 추천 결과는 "실험 검증된 결론"이 아니라 "모델 기반 가상 후보"로 표시한다.
- 외부 가격 DB, 논문 검색, 특허 검색, RDKit 도구는 agent layer에서 붙이고, ML training pipeline에 섞지 않는다.
