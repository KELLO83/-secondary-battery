# XAI and Monitoring

## Scope

NASA battery capacity prediction 모델의 설명 가능성, 신뢰도, drift/OOD 경고 정책을 정의한다.

## XAI Targets

설명 대상:

```text
predicted_capacity
estimated_soh
capacity_error
```

1차 XAI:

- LightGBM/CatBoost: SHAP TreeExplainer
- Neural/Transformer: permutation importance 또는 Captum 후보
- Foundation model: local perturbation 기반 설명

## Expected Explanations

예측마다 다음 정보를 제공한다.

```text
top_positive_features
top_negative_features
similar_reference_cycles
feature_distribution_warning
```

예:

```text
duration_sec와 voltage_measured_last가 capacity 예측을 낮추는 방향으로 크게 기여했다.
temperature_measured_mean은 학습 분포 중앙에 있어 OOD 경고는 없다.
```

## OOD Checks

기본 OOD 기준:

- numeric feature가 train quantile 0.5% ~ 99.5% 밖에 있는지
- validation에 없던 `battery_id` 또는 비정상 category인지
- duration/current/voltage 통계가 물리적으로 비정상인지

## Drift Monitoring

운영 시 저장할 분포:

```text
capacity prediction distribution
voltage/current/temperature feature distribution
battery_id/cycle_index distribution
prediction error distribution when ground truth arrives
```

## Alert Policy

경고 등급:

| Level | 조건 |
|---|---|
| low | 정상 분포 |
| medium | 일부 feature quantile 초과 |
| high | 여러 핵심 feature OOD 또는 예측 신뢰도 낮음 |

high 경고가 발생하면 agent는 실험 추천이 아니라 검토 요청 리포트를 생성한다.
