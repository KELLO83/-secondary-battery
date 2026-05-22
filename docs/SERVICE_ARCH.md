# SERVICE_ARCH: 서버, 프론트엔드, PostgreSQL 아키텍처

## 1. 목적

본 문서는 2차전지 정형 데이터 모델 실험 결과를 서비스화하고, 관리자 화면과 AI 에이전트 시스템을 구축하기 위한 서버/프론트엔드/DB 아키텍처를 정의한다.

ML 실험 파이프라인은 `docs/SYSTEM_ARCH.md`를 따르고, 서비스화 단계는 본 문서를 따른다.

## 2. 기술 스택

### 2.1 Frontend

권장:

- React
- TypeScript
- Vite
- TanStack Query
- TanStack Table
- Recharts 또는 ECharts

역할:

- 관리자 대시보드
- 실험 결과 leaderboard 조회
- 모델 버전/배포 상태 확인
- 단일 예측 입력 화면
- batch 후보 screening 화면
- AI 에이전트 대화 및 리포트 화면
- XAI/OOD 신뢰도 경고 표시

대안:

- Vue.js도 가능하지만, 본 프로젝트의 기본 프론트엔드 스택은 React로 둔다.

### 2.2 Backend

권장:

- FastAPI
- Pydantic
- SQLAlchemy
- Alembic
- Uvicorn

역할:

- REST API 제공
- 모델 artifact 로딩
- 예측 API 제공
- 실험 결과 조회
- 모델 registry 관리
- AI 에이전트 실행 API 제공
- PostgreSQL 연동

### 2.3 Database

기본 DB:

- PostgreSQL

역할:

- 실험 metadata 저장
- 모델 버전 저장
- 예측 요청/결과 저장
- AI 에이전트 실행 로그 저장
- 후보 조합 및 리포트 저장
- 사용자/관리자 계정 저장, 필요 시

### 2.4 Optional Infrastructure

초기 필수는 아니지만 후속 단계에서 검토한다.

- Redis: cache, async job queue
- Celery 또는 RQ: batch screening, agent long-running task
- MinIO 또는 S3: model artifact/object storage
- MLflow: model registry 및 experiment tracking
- Vector DB: 논문/문헌 RAG 검색

## 3. 권장 디렉터리 구조

```text
secanday_battery/
  backend/
    app/
      __init__.py
      main.py
      core/
        config.py
        logging.py
        security.py
      db/
        session.py
        base.py
        models.py
        migrations/
      schemas/
        experiment.py
        model.py
        prediction.py
        agent.py
      api/
        __init__.py
        routes/
          health.py
          experiments.py
          models.py
          predictions.py
          agent.py
      services/
        experiment_service.py
        model_service.py
        prediction_service.py
        agent_service.py
      ml/
        artifact_loader.py
        predictor.py
        preprocessing.py
      agent/
        planner.py
        candidate_generator.py
        tools.py
        report_writer.py
    alembic.ini
    requirements-314.txt
  frontend/
    src/
      app/
      api/
      components/
      pages/
        DashboardPage.tsx
        ExperimentsPage.tsx
        ModelsPage.tsx
        PredictionPage.tsx
        AgentPage.tsx
      routes/
      types/
    package.json
  results/
    model_artifacts/
```

서비스 코드와 ML 학습 코드는 분리한다.

원칙:

- `backend/`는 `.venv314` 환경에서 실행한다.
- `backend/`는 `ml/`의 학습 스크립트를 직접 실행하지 않는다.
- `backend/`는 `results/model_artifacts/`에 저장된 best model artifact와 preprocessing artifact를 로드해 예측만 수행한다.
- 모델 재학습, 대규모 batch 학습, leaderboard 생성은 `ml/`에서 수행한다.
- 프론트엔드는 `frontend/` 아래 React/TypeScript 프로젝트로 관리한다.

## 4. Backend API 설계

### 4.1 Health

```text
GET /health
```

목적:

- 서버 상태 확인
- DB 연결 상태 확인

### 4.2 Experiments

```text
GET /experiments
GET /experiments/{experiment_id}
POST /experiments/import
```

역할:

- `results/experiments.csv` 또는 JSONL 결과를 PostgreSQL로 import
- 실험 목록 조회
- leaderboard용 필터링 제공

필터 후보:

- model_name
- model_family
- split_type
- sample_size
- pretrained
- group_key

### 4.3 Models

```text
GET /models
GET /models/{model_version}
POST /models/register
POST /models/{model_version}/activate
```

역할:

- 학습된 model artifact 등록
- best model 지정
- active serving model 설정
- pretrained checkpoint 정보 확인

### 4.4 Predictions

```text
POST /predictions
GET /predictions/{prediction_id}
GET /predictions
```

역할:

- 단일 battery recipe 예측
- 예측 요청/결과 저장
- 사용한 model version 기록
- OOD 경고 기록
- XAI contribution 및 confidence score 반환, 구현 시

입력 예시:

```json
{
  "material_structure": "Layered",
  "synthesis_method": "Solid-state",
  "Li_source": "LiOH",
  "Ni_source": "NiSO4",
  "Co_source": "Co(NO3)2",
  "Mn_source": "MnCO3",
  "electrolyte": "example",
  "separator": "example",
  "counter_electrode": "Li-metal",
  "voltage_range_V_min": 2.5,
  "voltage_range_V_max": 4.3
}
```

출력 예시:

```json
{
  "prediction_id": "uuid",
  "predicted_remain_capacity": 2.85,
  "model_version": "catboost_v1",
  "is_out_of_distribution": false,
  "ood_score": 0.12,
  "confidence_score": 0.88,
  "explanation": {
    "method": "shap",
    "top_positive": [],
    "top_negative": []
  },
  "warning": null
}
```

API 요청 필드명은 JSON 호환성과 프론트엔드 사용성을 위해 snake_case를 사용한다. 백엔드는 이 값을 `docs/DATASETEXPLAIN.md`의 원본 칼럼명으로 매핑한 뒤 모델에 전달한다.

필수 매핑:

```text
voltage_range_V_min -> voltage_range(V)_min
voltage_range_V_max -> voltage_range(V)_max
```

그 외 범주형 feature는 원본 칼럼명과 같은 이름을 사용한다.

### 4.5 Agent

```text
POST /agent/runs
GET /agent/runs/{run_id}
GET /agent/runs
```

역할:

- 자연어 설계 목표를 받아 AI 에이전트 실행
- 후보 조합 생성
- `capacity_predictor_tool` 호출
- 추천 후보와 리포트 저장

주의:

- agent 결과는 "예측 기반 후보"로 표시한다.
- 실험 검증된 결론으로 표현하지 않는다.

## 5. PostgreSQL 테이블 설계

### 5.1 `experiments`

실험 결과 metadata 저장.

주요 컬럼:

```text
id UUID primary key
experiment_id TEXT unique
model_name TEXT
model_family TEXT
pretrained BOOLEAN
checkpoint TEXT
weight_source TEXT
access_mode TEXT
license_checked BOOLEAN
data_size TEXT
sample_size BIGINT
split_type TEXT
split_seed INTEGER
group_key TEXT
features_used JSONB
target_transform TEXT
preprocessing_version TEXT
hyperparameters JSONB
train_time_sec DOUBLE PRECISION
predict_time_sec DOUBLE PRECISION
peak_memory_mb DOUBLE PRECISION
gpu_memory_mb DOUBLE PRECISION
valid_mape DOUBLE PRECISION
valid_mae DOUBLE PRECISION
valid_rmse DOUBLE PRECISION
test_mape DOUBLE PRECISION
test_mae DOUBLE PRECISION
test_rmse DOUBLE PRECISION
artifact_path TEXT
notes TEXT
created_at TIMESTAMPTZ
```

### 5.2 `model_versions`

서빙 가능한 모델 버전 저장.

주요 컬럼:

```text
id UUID primary key
model_version TEXT unique
experiment_id TEXT
model_name TEXT
model_family TEXT
artifact_path TEXT
preprocessing_path TEXT
schema_version TEXT
is_active BOOLEAN
is_best BOOLEAN
pretrained BOOLEAN
checkpoint TEXT
metrics JSONB
created_at TIMESTAMPTZ
activated_at TIMESTAMPTZ
```

### 5.3 `prediction_requests`

단일 예측 요청 및 결과 저장.

주요 컬럼:

```text
id UUID primary key
model_version TEXT
input_recipe JSONB
predicted_remain_capacity DOUBLE PRECISION
is_out_of_distribution BOOLEAN
ood_score DOUBLE PRECISION
confidence_score DOUBLE PRECISION
explanation JSONB
warning TEXT
latency_ms DOUBLE PRECISION
created_at TIMESTAMPTZ
```

### 5.4 `agent_runs`

AI 에이전트 실행 단위 저장.

주요 컬럼:

```text
id UUID primary key
user_goal TEXT
parsed_constraints JSONB
planner_trace JSONB
status TEXT
model_version TEXT
final_report TEXT
created_at TIMESTAMPTZ
finished_at TIMESTAMPTZ
```

### 5.5 `agent_candidates`

에이전트가 생성한 후보 조합 및 평가 결과 저장.

주요 컬럼:

```text
id UUID primary key
agent_run_id UUID
candidate_recipe JSONB
predicted_remain_capacity DOUBLE PRECISION
rank INTEGER
score DOUBLE PRECISION
is_out_of_distribution BOOLEAN
evidence JSONB
warning TEXT
created_at TIMESTAMPTZ
```

### 5.6 `users`, 선택

초기 내부 도구에서는 생략 가능하다.

필요 시 관리자 인증을 위해 추가한다.

## 6. Artifact 저장 정책

PostgreSQL에는 모델 파일 자체를 저장하지 않는다.

DB에는 경로와 metadata만 저장한다.

```text
results/model_artifacts/{experiment_id}/
  model.*
  preprocessing.*
  config.yaml
  metrics.json
  feature_info.json
```

후속 운영 환경에서는 local path 대신 S3 또는 MinIO object path를 사용한다.

## 7. 서비스 구축 단계

### Phase S1: Backend + DB 기본 구조

- FastAPI 프로젝트 생성
- PostgreSQL 연결
- SQLAlchemy model 작성
- Alembic migration 설정
- `/health` API 구현

### Phase S2: Experiment Dashboard API

- `results/experiments.csv` import
- `experiments` table 저장
- leaderboard 조회 API 구현

### Phase S3: Model Registry API

- model artifact 등록
- active model 지정
- model metadata 조회

### Phase S4: Prediction API

- best model artifact 로딩
- preprocessing pipeline 로딩
- `/predictions` API 구현
- prediction log 저장

### Phase S5: Frontend Admin

- React 관리자 화면 구축
- leaderboard page
- model registry page
- prediction input page

### Phase S6: AI Agent API

- `docs/AGENT_SYSTEM.md` 기준 agent workflow 구현
- 후보 생성
- `capacity_predictor_tool` 연결
- agent run/candidate/report 저장

### Phase S7: Optional External Tools

- 가격 DB/API 연동
- 논문/특허 검색 연동
- vector DB/RAG 연동

## 8. 보안 및 운영 주의사항

- DB credential은 `.env`로 관리한다.
- API key와 외부 DB token은 코드에 하드코딩하지 않는다.
- model artifact path는 허용된 디렉터리 아래로 제한한다.
- agent 결과는 실험 검증된 사실처럼 표시하지 않는다.
- 관리자 기능은 추후 인증을 붙인다.

## 9. 초기 비범위

초기 서비스화 단계에서는 다음을 필수로 구현하지 않는다.

- 사용자 권한 체계 전체
- 대규모 비동기 batch job queue
- 외부 가격/공급망 API
- vector DB 기반 RAG
- 모델 자동 재학습
- Kubernetes 배포
