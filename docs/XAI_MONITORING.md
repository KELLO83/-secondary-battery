# XAI_MONITORING: 설명 가능한 AI 및 모델 신뢰성 가이드

## 1. 목적

본 문서는 2차전지 `remain_capacity` 예측 모델과 AI 에이전트 시스템의 설명 가능성, 신뢰도 경고, 데이터 드리프트 모니터링 정책을 정의한다.

배터리 연구원이 모델의 예측값을 실험 후보 선정에 참고하려면 단순 수치 예측뿐 아니라 다음 정보가 함께 제공되어야 한다.

- 왜 해당 예측값이 나왔는가
- 어떤 feature가 예측을 올리거나 낮췄는가
- 입력 조합이 학습 데이터 분포 안에 있는가
- 모델 예측을 어느 정도 신뢰할 수 있는가
- 운영 중 입력 분포나 예측 분포가 변하고 있는가

## 2. 핵심 원칙

- XAI 결과는 실험적 인과 증명이 아니라 모델 예측 해석이다.
- SHAP, LIME, permutation importance 등은 "모델이 어떤 근거로 예측했는지"를 설명할 뿐, 실제 화학적 원인을 증명하지 않는다.
- 모든 추천 결과에는 "예측 기반 후보"와 "실험 검증 필요" 문구를 유지한다.
- OOD 또는 drift 경고가 있는 후보는 에이전트 추천 순위에서 penalty를 받는다.

## 3. XAI 대상 모델

### 3.1 GBDT 계열

대상:

- LightGBM
- CatBoost

권장 방법:

- SHAP TreeExplainer
- built-in feature importance
- permutation importance

우선순위:

1. SHAP local explanation
2. SHAP global summary
3. permutation importance
4. model native importance

### 3.2 Neural 계열

대상:

- RealMLP
- TabM
- DCN-V2
- FT-Transformer
- TabTransformer
- TabNet

권장 방법:

- permutation importance
- integrated gradients, 구현 가능 시
- feature ablation
- surrogate GBDT 또는 surrogate linear model
- attention weight는 보조 지표로만 사용

주의:

- attention weight를 곧바로 과학적 중요도로 해석하지 않는다.
- neural model의 설명은 GBDT SHAP보다 불안정할 수 있으므로 리포트에 설명 방식과 한계를 명시한다.

### 3.3 Foundation Model 계열

대상:

- TabPFN-3
- TabICLv2

권장 방법:

- permutation importance
- feature ablation
- local surrogate model

주의:

- pretrained/foundation model의 내부 구조 설명이 제한될 수 있다.
- API 기반 모델은 원본 gradient나 internal attention 접근이 불가능할 수 있으므로 black-box explanation을 기본으로 한다.

## 4. Local Explanation

단일 예측 요청에 대해 feature별 기여도를 제공한다.

예시 출력:

```text
predicted_remain_capacity: 2.85
base_value: 2.63

feature_contributions:
  Li_source = LiOH: +0.12
  Ni_source = NiSO4: +0.08
  voltage_range(V)_max = 4.3: +0.06
  synthesis_method = Solid-state: -0.04
```

필수 표시:

- predicted value
- base value 또는 reference value
- positive contribution top-K
- negative contribution top-K
- explanation method
- model version

## 5. Global Explanation

모델 전체에서 중요한 feature를 요약한다.

권장 화면:

- SHAP summary plot
- feature importance bar chart
- target bin별 error 분석
- feature value별 partial dependence 또는 accumulated local effects, 가능 시

활용:

- 모델이 어떤 변수를 주로 사용하는지 확인
- leakage 의심 feature 탐지
- 연구원에게 모델의 전반적 경향 설명

## 6. OOD 감지 정책

OOD는 입력 recipe가 학습 데이터 분포에서 벗어났는지 판단하는 신뢰도 계층이다.

### 6.1 OOD 신호

범주형:

- 학습 데이터에 없던 category
- 드문 category 조합
- 특정 source 조합의 train support 부족

수치형:

- `voltage_range(V)_min/max`가 학습 데이터 min/max 밖
- 학습 분포의 극단 quantile 밖

조합:

- 전체 feature 조합과 유사한 train sample이 거의 없음
- nearest-neighbor distance가 큼
- category novelty score가 높음

### 6.2 권장 OOD 지표

```text
unknown_category_count
rare_category_count
numeric_range_violation_count
nearest_neighbor_distance
train_support_count
category_novelty_score
ood_score
is_out_of_distribution
```

### 6.3 OOD 경고 등급

```text
LOW:
  학습 데이터 분포 안에 있는 일반 입력

MEDIUM:
  일부 드문 category 또는 낮은 support 조합

HIGH:
  unknown category, 수치형 범위 초과, 매우 낮은 train support
```

## 7. Prediction Confidence

초기 confidence는 calibration된 확률이 아니라 heuristic reliability score로 정의한다.

초기 산정 요소:

- OOD score
- train support count
- validation error by target bin
- model family reliability
- random split vs group split 성능 차이

예시:

```text
confidence_score = 1.0
confidence_score -= ood_penalty
confidence_score -= low_support_penalty
confidence_score -= group_split_gap_penalty
```

주의:

- confidence score는 실제 성공 확률이 아니다.
- 실험 결과가 누적되면 calibration을 별도 수행한다.

## 8. Drift Monitoring

서비스 운영 중 입력 및 예측 분포 변화를 감지한다.

감시 대상:

- feature별 입력 분포
- 신규 category 발생률
- OOD 비율
- predicted_remain_capacity 분포
- model version별 예측 차이
- 실제 실험 결과가 들어올 경우 prediction error

권장 지표:

```text
population_stability_index
kl_divergence
new_category_rate
ood_rate
prediction_mean_shift
prediction_std_shift
error_mape_recent
```

## 9. API 응답 확장 필드

`/predictions` API는 다음 XAI/신뢰도 필드를 포함할 수 있다.

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
    "base_value": 2.63,
    "top_positive": [
      {"feature": "Li_source", "value": "LiOH", "contribution": 0.12}
    ],
    "top_negative": [
      {"feature": "synthesis_method", "value": "Solid-state", "contribution": -0.04}
    ]
  },
  "warning": null
}
```

## 10. 관리자 화면 표시 항목

Prediction 화면:

- 예측 remain_capacity
- confidence score
- OOD 등급
- top positive contributions
- top negative contributions
- warning

Model 화면:

- global feature importance
- SHAP summary, 가능 시
- random/group split metric
- 최근 OOD 비율
- 최근 예측 분포 변화

Agent 화면:

- 후보별 예측값
- 후보별 confidence score
- 후보별 OOD 경고
- 후보별 주요 기여 feature
- 추천 제외 또는 penalty 사유

## 11. 에이전트 연계 규칙

AI 에이전트는 후보 ranking 시 XAI/신뢰도 정보를 사용한다.

규칙:

- OOD HIGH 후보는 기본적으로 추천 우선순위를 낮춘다.
- confidence score가 낮은 후보는 "실험 검증 우선순위 낮음" 또는 "추가 검증 필요"로 표시한다.
- top contribution은 추천 이유 설명에 사용하되, 과학적 인과로 단정하지 않는다.
- 후보 리포트에는 예측값, 기여 feature, OOD 경고, 한계를 함께 표시한다.

## 12. 저장 데이터

PostgreSQL 또는 artifact JSON에 다음 정보를 저장한다.

```text
prediction_id
model_version
explanation_method
base_value
feature_contributions
ood_score
ood_level
confidence_score
drift_snapshot_version
warning
created_at
```

대용량 SHAP 배열은 DB에 직접 저장하지 않고 JSON artifact 또는 object storage에 저장할 수 있다.

## 13. 구현 단계

### Phase X1: 기본 신뢰도 필드

- unknown category 감지
- numeric range violation 감지
- train support count 계산
- `is_out_of_distribution` 반환

### Phase X2: GBDT SHAP

- LightGBM/CatBoost SHAP local explanation
- top positive/negative contribution 반환
- 관리자 화면 표시

### Phase X3: Global Explanation

- global importance 생성
- SHAP summary artifact 저장
- leaderboard와 모델 화면 연결

### Phase X4: Drift Monitoring

- prediction log 기반 입력 분포 추적
- OOD rate 추적
- prediction distribution shift 추적

### Phase X5: Agent Integration

- confidence score를 agent ranking에 반영
- agent report에 XAI와 OOD 경고 포함

## 14. 비범위

초기 구현에서는 다음을 필수로 하지 않는다.

- XAI 결과를 실제 화학적 인과로 주장
- 모든 neural/foundation model에 gradient 기반 설명 강제
- 실시간 고비용 SHAP 계산
- 완전한 uncertainty quantification
- 실험 성공 확률 calibration
