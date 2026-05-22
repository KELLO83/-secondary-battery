# AGENT: AI 코딩 에이전트 행동 강령

## 1. 목적

본 문서는 2차전지 정형 데이터 실험 코드를 작성하는 AI 에이전트의 개발 규칙을 정의한다.

AI는 코드를 작성하기 전에 다음 문서를 우선순위대로 확인해야 한다.

1. `docs/PRD.md`: 목표, 성공 기준, 산출물
2. `docs/DATASETEXPLAIN.md`: 데이터 스키마, feature/target, 제외 칼럼
3. `docs/SYSTEM_ARCH.md`: 디렉터리 구조, 모듈 책임, 인터페이스
4. `docs/MODEL_TIERS.md`: 모델별 실험 규모와 hyperparameter 범위
5. `AGENT.md`: 코드 스타일과 검증 규칙

## AGENT_SYSTEM.md 설명

`docs/AGENT_SYSTEM.md`는 코딩 에이전트 행동 규칙 문서가 아니라, 학습된 2차전지 예측 모델을 활용하는 배터리 설계 AI 에이전트 시스템의 기획 문서이다.

이 문서는 최종 예측 모델을 `capacity_predictor_tool`로 연결해 자연어 설계 목표를 해석하고, 후보 조합 생성, 가상 예측, 필터링, 랭킹, 리포트 생성을 수행하는 후속 Phase 6 시스템을 설명한다.

코드 작성 에이전트는 `docs/AGENT_SYSTEM.md`를 ML 학습 파이프라인의 직접 구현 규칙으로 사용하지 않고, Phase 5 이후 에이전트 시스템을 설계하거나 예측 모델을 tool로 노출할 때 참고한다.

## SERVICE_ARCH.md 설명

`docs/SERVICE_ARCH.md`는 학습된 모델과 실험 결과를 실제 서버/관리자 화면/AI 에이전트 서비스로 연결하기 위한 서비스 아키텍처 문서이다.

이 문서는 React 프론트엔드, FastAPI 백엔드, PostgreSQL 데이터베이스를 기본 스택으로 정의하며, 실험 결과 조회, 모델 버전 관리, 예측 API, AI 에이전트 실행 로그 저장, 후보 조합 리포트 저장 구조를 설명한다.

코드 작성 에이전트는 순수 ML 실험 파이프라인을 구현할 때는 `docs/SYSTEM_ARCH.md`와 `docs/MODEL_TIERS.md`를 우선 따르고, 서버/API/DB/관리자 화면을 구축하는 단계에서는 `docs/SERVICE_ARCH.md`를 참조한다.

## 코드 디렉터리 경계

코드는 목적에 따라 디렉터리를 분리한다.

```text
ml/
  모델 학습, 전처리, 평가, 실험 runner, leaderboard

backend/
  FastAPI 서버, PostgreSQL 연동, model artifact 로딩, prediction API, agent API

frontend/
  React 관리자 화면, 예측 UI, AI 에이전트 UI
```

규칙:

- ML 모델 학습 코드는 `ml/` 아래에 작성한다.
- 서버/API 코드는 `backend/` 아래에 작성한다.
- 프론트엔드 코드는 `frontend/` 아래에 작성한다.
- `backend/`는 모델을 재학습하지 않고 `results/model_artifacts/`의 저장된 모델과 preprocessing artifact를 로드한다.
- ML 실행 환경은 단순하게 구분한다.
- LightGBM만 `.venv314t`에서 실행한다.
- LightGBM을 제외한 모든 모델은 `.venv314`에서 실행한다.
- CatBoost GPU baseline은 `.venv314`에서 실행한다.
- Neural/Transformer/Foundation 모델은 GPU 학습 또는 GPU 추론을 기본으로 하므로 `.venv314`에서 실행한다.
- `backend/`는 `.venv314`, `frontend/`는 Node.js 패키지 환경을 사용한다.
- TabICLv2, TabPFN 계열처럼 pretrained/in-context 방식으로 동작하는 모델은 프로젝트 데이터로 from-scratch 학습하지 않고, training-free 또는 inference-only 실험으로 분리 기록한다.
- 외부 pip/오픈소스 구현이 있는 Neural/Transformer/Foundation 모델은 프로젝트 내부에 모델 본체를 재구현하지 않고, 모델별 wrapper 파일에서 import해서 사용한다.
- 모델 wrapper는 모델별 파일로 분리한다. 예: `TabPFN.py`에는 TabPFN 관련 코드만 둔다.

## XAI_MONITORING.md 설명

`docs/XAI_MONITORING.md`는 예측 모델과 AI 에이전트의 설명 가능성, OOD 감지, confidence score, drift monitoring 정책을 정의하는 문서이다.

이 문서는 SHAP, permutation importance, local explanation, global feature importance, 데이터 분포 밖 입력 경고, 운영 중 drift 감지를 다룬다.

코드 작성 에이전트는 예측 API, 관리자 화면, AI 에이전트 리포트에 설명 가능성과 신뢰도 경고를 추가하는 단계에서 `docs/XAI_MONITORING.md`를 참조한다.

## DATA_CONTRACT.md 설명

`docs/DATA_CONTRACT.md`는 `docs/DATASETEXPLAIN.md`의 데이터 설명을 코드에서 검증 가능한 입력 계약으로 정리한 문서이다.

이 문서는 필수 칼럼, target 범위, metadata 제외 규칙, categorical unknown 처리, voltage field 매핑, validation 실패 유형을 정의한다.

코드 작성 에이전트는 데이터 로더, 전처리 파이프라인, FastAPI 예측 request schema를 구현할 때 `docs/DATA_CONTRACT.md`를 참조한다.

## EVALUATION_PROTOCOL.md 설명

`docs/EVALUATION_PROTOCOL.md`는 모델 성능을 공정하게 비교하기 위한 평가 규칙 문서이다.

이 문서는 RMSE/MAE/WAPE/SMAPE/filtered MAPE/raw MAPE 계산, random split과 group split 분리, 동일 sample/split 조건 비교, leaderboard 정렬 기준, 최종 모델 선정 기준을 정의한다.

코드 작성 에이전트는 metric 계산, leaderboard 생성, 실험 결과 비교 로직을 구현할 때 `docs/EVALUATION_PROTOCOL.md`를 참조한다.

## Python 실행 환경 정책

모델 훈련 환경과 일반 서버 실행 환경은 분리한다.

### ML 모델 훈련 환경

CPU 중심 ML 모델 훈련과 대규모 전처리는 다음 가상환경을 사용한다.

```text
.venv314t
```

목적:

- Python 3.14t free-threaded, GIL 해제 버전 사용
- CPU 기반 ML 훈련 시 멀티스레드 활용 극대화
- LightGBM CPU baseline 실험용 기본 환경
- pandas/scikit-learn/LightGBM 등 CPU multi-thread 작업용 환경

주의:

- 라이브러리가 Python 3.14t 또는 free-threaded runtime을 지원하지 않는 경우, 해당 모델은 호환 가능한 환경에서 별도 실행하고 사유를 기록한다.
- `.venv314t`에서 pip install 또는 wheel 로딩이 실패하는 패키지는 무리하게 우회하지 않고, `.venv314` 또는 해당 패키지가 공식 지원하는 별도 Python 환경에서 실행한 뒤 실행 환경 차이를 실험 로그에 기록한다.
- `.venv314t`에서 호환 wheel이 없다는 이유로 패키지를 직접 소스 빌드하지 않는다. 특히 PyTorch, LightGBM, CatBoost, CUDA 관련 패키지는 사용자가 명시적으로 지시하지 않는 한 소스 빌드를 시도하지 않는다.
- 모델별 실행 환경은 실험 로그에 기록한다.

CatBoost 예외:

- CatBoost는 Python 3.14t/free-threaded 공식 wheel이 없으면 `.venv314t`에 설치하지 않는다.
- CatBoost baseline은 일반 Python 3.14 환경인 `.venv314`에서 공식 PyPI wheel로 실행한다.
- `PYTHON_GIL=1`은 wheel ABI 문제를 해결하지 못하므로 CatBoost 설치 우회책으로 사용하지 않는다.
- CatBoost 메인 baseline은 `.venv314`에서 GPU를 기본값으로 실행한다.
- CatBoost GPU 실행은 `task_type=GPU`, `devices=0`, `gpu_ram_part=0.90`을 기본 정책으로 사용한다.
- CatBoost CPU 실행은 smoke/debug/fallback 용도로만 사용하며, 필요한 경우 `--task-type CPU`를 명시한다.
- CatBoost CPU fallback은 내부 C++ 멀티스레딩을 사용하므로 `.venv314`에서 `thread_count=14`로 실행한다.
- CatBoost 결과는 `python_executable`, `python_version`, `thread_count`를 실험 로그에 남겨 LightGBM 결과와 실행 환경 차이를 추적한다.

Neural/Transformer/Foundation 예외:

- RealMLP, TabM, TabR, DCN-V2, NODE, FT-Transformer, TabTransformer, TabNet, TabPFN, TabPFN latest, TabICLv2, AutoGluon/Mitra는 기본적으로 GPU 학습 또는 GPU 추론을 사용한다.
- 이 모델들은 PyTorch/CUDA/공식 pretrained checkpoint 호환성이 중요하므로 `.venv314`에서 설치하고 실행한다.
- `.venv314t`에 PyTorch, CUDA extension, tabular foundation model package를 소스 빌드해서 맞추지 않는다.
- `ml/requirements-optional-dl.txt`는 `.venv314` 기준 optional dependency 목록으로 취급한다.
- GPU를 사용할 수 없는 경우에만 명시적으로 CPU fallback을 검토하고, 실행 환경과 사유를 실험 로그에 기록한다.
- TabPFN pretrained weight는 Prior Labs 라이선스 승인과 token 또는 local checkpoint가 필요할 수 있다. 정식 실험 스크립트는 브라우저 로그인 프롬프트에 의존하지 않고 `TABPFN_TOKEN`, `~/.cache/tabpfn/auth_token`, `~/.tabpfn/token`, 또는 명시적 local `model_path`를 사용한다.
- 대형 실험 실행 전에는 `docs/RUN_RISK_CHECKLIST.md`를 확인하고, GPU/OOM, TabPFN 접근권한, AutoGluon/Mitra 설치 조건을 충족하지 못하면 실패 사유를 기록한다.
- 모델 아키텍처 본체는 가능한 한 공식 GitHub/Hugging Face/PyPI/PyTorch 구현체를 import해서 사용한다. 프로젝트 내부 코드는 데이터 전처리, 학습 orchestration, logging, evaluation wrapper를 담당한다.
- 외부 구현체가 없거나 회귀/Python/CUDA 환경을 지원하지 않아 직접 구현이 필요한 경우, 코드 작성 전에 해당 사유를 문서에 남긴다.

### 서버/API/관리자 화면 실행 환경

FastAPI 백엔드, 관리자 API, 일반 서버 실행은 다음 가상환경을 사용한다.

```text
.venv314
```

목적:

- 일반 Python 3.14 환경 사용
- FastAPI, SQLAlchemy, Alembic, PostgreSQL 연동, 관리자 API 실행
- ML 훈련 환경과 서비스 runtime dependency 충돌 방지

프론트엔드는 Python 가상환경이 아니라 `frontend/`의 Node.js 패키지 환경을 사용한다.

## ML 훈련 리소스 사용 정책

## ML/AI 진행률 표시 정책

ML/AI 모델 개발 코드는 장시간 실행되는 작업의 진행 상황을 사용자가 확인할 수 있게 구성한다.

적용 대상:

- 대용량 CSV 로딩 및 chunk sampling
- 전처리, encoding, split 생성
- 모델 학습 epoch/iteration loop
- batch prediction
- hyperparameter sweep
- foundation model sample 평가
- leaderboard 생성

원칙:

- 가능한 경우 `tqdm` 기반 progress bar를 사용한다.
- LightGBM, CatBoost처럼 자체 logging/callback이 있는 모델은 progress bar 또는 주기적 log 중 하나를 제공한다.
- CLI script는 `--no-progress` 옵션을 둘 수 있으나 기본값은 progress 표시이다.
- progress 출력은 metric/result CSV를 오염시키지 않는다.
- 긴 작업은 현재 단계, 처리 row 수, 전체 row 수 또는 chunk 수, elapsed time을 알 수 있게 한다.
- 서버/API runtime에서는 terminal progress bar 대신 structured log 또는 DB job status로 진행 상태를 노출한다.

## ML/AI 터미널 로깅 정책

ML/AI 모델 개발 코드는 중요한 실행 상태를 logger의 `info` 레벨로 터미널에 출력한다.

적용 대상:

- 데이터 파일 로딩 시작/완료
- 사용 feature set, target, metadata 제외 칼럼
- train/validation row 수
- sample size, seed, source_family별 row 수
- 모델명, 실행 환경, Python executable/version
- LightGBM/CatBoost 주요 hyperparameter
- CatBoost GPU 실행 여부, `task_type`, `devices`, `gpu_ram_part`
- GPU/CPU worker 설정
- 학습 시작/완료, 학습 시간
- 예측 시작/완료, 예측 시간
- metric 계산 결과
- 결과 CSV/JSON 저장 경로
- model artifact 저장/로드 경로, 구현 시

원칙:

- 단순 `print()` 대신 표준 `logging` 모듈 또는 프로젝트 공통 logger를 사용한다.
- 기본 로그 레벨은 `INFO`로 둔다.
- 경고성 상황은 `warning`, 실패는 `error` 또는 예외로 남긴다.
- progress bar와 logger 출력이 서로 깨지지 않도록 `tqdm.write()` 또는 logging handler 설정을 사용한다.
- 실험 결과 CSV에는 metric/config를 저장하고, 터미널 log는 실행 상태를 사람이 추적하기 위한 용도로 둔다.

## ML/AI 실험 실행 단위 정책

모델 훈련 스크립트는 한 번 실행할 때 정확히 하나의 실험만 수행한다.

원칙:

- 1 script run = 1 model = 1 feature set = 1 sample/full-data setting = 1 seed = 1 training job
- 하나의 CLI 실행에서 여러 모델을 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 sample size를 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 feature set을 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 seed 또는 hyperparameter sweep을 순차 실행하지 않는다.
- sweep, grid search, AutoML, batch leaderboard runner를 만들지 않는다.
- leaderboard 생성은 이미 완료된 `results/experiments.csv`를 읽는 후처리 작업으로만 수행한다.
- 대용량 학습은 사용자가 명시적으로 실행한 단일 명령만 수행한다.
- 실패한 실험을 자동으로 다음 실험으로 넘어가게 만들지 않는다.
- 재시도도 사용자의 명시적 명령으로만 수행한다.

이유:

- 16M row급 데이터에서 연속 실험은 GPU/CPU/메모리 점유 위험이 크다.
- 여러 실험이 한 프로세스에 섞이면 로그, 진행률, 실패 원인, 결과 CSV 해석이 어려워진다.
- CatBoost GPU 학습처럼 장시간/고자원 작업은 사용자가 실험 단위를 명확히 통제해야 한다.

### GPU 훈련

GPU 사용 가능 모델은 가능한 한 GPU를 우선 사용한다.

기본 정책:

```text
target_gpu_memory_utilization: 0.90
```

원칙:

- 모델 훈련 시 GPU 가용 메모리의 최대 90%까지 사용하는 것을 목표로 한다.
- Neural/Transformer/Foundation 모델 학습 코드는 가능한 경우 `auto_batch` 또는 batch size probing을 구현해 GPU memory 사용량이 `target_gpu_memory_utilization=0.90`에 근접하도록 batch size를 자동 선택한다.
- auto batch는 작은 batch에서 시작해 단계적으로 키우고, OOM 또는 CUDA memory error가 발생하면 직전 성공 batch size로 되돌린다.
- auto batch probing 후에는 선택된 `batch_size`, 추정/실측 GPU memory 사용량, probing 실패 여부를 logger `info`와 실험 결과에 기록한다.
- 사용자가 `--batch-size`를 명시한 경우에는 해당 값을 우선하되, OOM 발생 시 자동 축소 여부와 최종 batch size를 로그에 남긴다.
- Foundation/in-context 모델처럼 batch size 의미가 일반 학습과 다른 경우에는 해당 패키지가 제공하는 `batch_size`, `eval_batch_size`, `memory_saving_mode`, `offload_mode`, `kv_cache` 옵션을 사용해 GPU memory를 최대한 활용하되, pretrained weight 자체를 fine-tuning하지 않는 모델은 강제로 학습 batch를 키우지 않는다.
- OOM이 발생하면 batch size, embedding dimension, model depth 순서로 줄인다.
- GPU memory 사용량은 실험 결과에 기록한다.
- 다른 서비스 프로세스가 같은 GPU를 사용 중이면 90% 정책을 낮출 수 있다.

### CPU 훈련

CPU 기반 LightGBM 훈련만 `.venv314t` 환경에서 실행한다. LightGBM을 제외한 모든 모델은 `.venv314`에서 실행한다.

기본 정책:

```text
cpu_workers: 14
```

원칙:

- CPU 기반 LightGBM, CatBoost, 전처리, 샘플링, batch prediction 작업은 기본 14 workers를 사용한다.
- 라이브러리별 thread 파라미터가 있으면 14로 맞춘다.
  - LightGBM: `num_threads=14`
  - CatBoost: `thread_count=14`
  - joblib/sklearn: `n_jobs=14`
  - PyTorch DataLoader: `num_workers`는 데이터/OS 안정성을 보고 최대 14까지 사용
- 시스템 부하가 과도하거나 메모리 병목이 발생하면 workers를 낮추고 사유를 기록한다.
