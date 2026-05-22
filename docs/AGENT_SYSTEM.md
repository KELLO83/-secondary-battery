# AI Agent System

## Scope

AI agent는 NASA battery cycle-level 모델을 도구로 사용해 배터리 열화 분석, capacity 추정, SOH/RUL 후보 분석을 보조한다.

## Core Tool

```text
input: cycle metadata + discharge summary features
model output: predicted_capacity
optional output: estimated_soh, degradation_risk
```

## Primary Workflows

1. 특정 배터리 cell의 cycle별 capacity degradation 요약
2. 새 discharge cycle summary 입력 시 capacity 예측
3. validation battery와 유사한 degradation pattern 검색
4. 예측 신뢰도와 OOD 경고 제공
5. 실험 리포트 초안 생성

## Tool Calling Policy

Agent는 직접 모델을 재학습하지 않는다.

허용:

- trained model inference
- XAI/SHAP summary 조회
- historical cycle 검색
- 리포트 생성

금지:

- 사용자 승인 없는 재학습
- leaderboard metric 조작
- `battery_id` group split 정책 위반

## Example Request

```text
B0005와 유사한 열화 패턴을 보이는 battery를 찾고,
cycle 80 이후 capacity 하락 위험을 요약해줘.
```

## Output Fields

```text
predicted_capacity
estimated_soh
nearest_reference_batteries
confidence_level
ood_warning
explanation_summary
```
