# Service Architecture

## Stack

- Frontend: React or Vue.js
- Backend: FastAPI
- DB: PostgreSQL
- ML runtime: Python `.venv314`
- CPU-only LightGBM/free-threaded experiments may use `.venv314t` when package support is available.

## Service Modules

```text
apps/frontend/
apps/api/
ml/
data/
results/
```

## Backend Responsibilities

- model registry 조회
- capacity prediction API
- experiment result 조회
- XAI summary 조회
- OOD/drift warning 제공
- PostgreSQL persistence

## Core API

```text
POST /api/predict/capacity
GET  /api/models
GET  /api/experiments
GET  /api/batteries/{battery_id}/cycles
POST /api/agent/analyze
```

## Prediction Response

```json
{
  "predicted_capacity": 1.42,
  "estimated_soh": 0.83,
  "model_name": "lightgbm",
  "feature_set": "discharge_summary",
  "confidence_level": "medium",
  "ood_warning": false
}
```

## PostgreSQL Tables

```text
experiments
models
battery_cycles
predictions
agent_reports
```

## Admin UI

관리자 화면은 다음을 보여준다.

- dataset summary
- battery별 cycle/capacity curve
- experiment leaderboard
- model artifact status
- prediction/XAI logs
- OOD/drift alerts
