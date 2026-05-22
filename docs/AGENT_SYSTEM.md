# AGENT_SYSTEM: 2차전지 설계 AI 에이전트 기획

## 1. 목적

본 문서는 `docs/PRD.md`의 Phase 6 후속 과제로, 학습된 2차전지 `remain_capacity` 예측 모델을 활용하는 배터리 설계 AI 에이전트 시스템의 목적, workflow, tool 구성, 안전 규칙을 정의한다.

이 문서의 에이전트는 코딩 에이전트가 아니라 배터리 연구원을 보조하는 가상 연구 조수이다.

## 2. 핵심 개념

단순 ML 모델:

```text
11개 입력 feature -> remain_capacity 예측값
```

AI 에이전트 시스템:

```text
연구 목표/제약 조건
-> 후보 조합 생성
-> ML 예측 모델 반복 호출
-> 조건 만족 후보 필터링
-> 비용/문헌/공급망/도메인 규칙 결합
-> 추천 리포트 생성
```

즉, ML 모델은 에이전트가 사용하는 하위 도구이며, 에이전트는 목표 기반 탐색과 의사결정 지원을 담당한다.

## 3. 사용 시나리오

예시 입력:

```text
양극재 결정 구조는 Layered로 유지하고,
remain_capacity가 2.8 Ah 이상으로 예측되는 조합 중
비용이 낮고 공급 안정성이 좋은 Li/Ni/Co/Mn source 후보를 추천해줘.
```

에이전트 처리 순서:

1. 자연어 목표 해석
2. 고정 조건과 최적화 목표 분리
3. 후보 조합 생성
4. `capacity_predictor_tool` 반복 호출
5. 목표 성능 미달 후보 제거
6. 비용/공급망/문헌 근거 결합
7. 후보 Top-N 정렬
8. XAI/OOD/신뢰도 경고 결합
9. 추천 리포트 작성

## 4. Tool 구성

### 4.1 필수 Tool

#### `capacity_predictor_tool`

역할:

- 학습된 best model artifact를 사용해 후보 조합의 `remain_capacity`를 예측한다.

입력:

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

출력:

```text
predicted_remain_capacity
model_name
model_version
validation_summary
is_out_of_distribution
ood_score
confidence_score
top_feature_contributions
warning
```

### 4.2 선택 Tool

후속 단계에서 추가한다.

- `cost_lookup_tool`: 원료 가격 조회
- `supply_chain_tool`: 공급 안정성 조회
- `literature_search_tool`: 논문 근거 검색
- `patent_search_tool`: 특허 중복성 탐색
- `formula_parser_tool`: 화학식 parsing
- `rdkit_tool`: 가능한 경우 화학 구조/물성 보조 계산

## 5. 에이전트 Workflow

```text
User Goal
  -> Planner
  -> Constraint Parser
  -> Candidate Generator
  -> Capacity Predictor Tool
  -> Filter
  -> Ranker
  -> Evidence Collector
  -> Report Writer
```

### 5.1 Planner

역할:

- 사용자의 자연어 목표를 정형화한다.
- 최적화 목표와 제약 조건을 분리한다.

예시:

```text
fixed:
  material_structure: Layered
objective:
  maximize: remain_capacity
constraints:
  remain_capacity >= 2.8
  cost: low
```

### 5.2 Candidate Generator

역할:

- 학습 데이터에 존재하는 category vocabulary를 기반으로 후보 조합을 생성한다.
- 학습 데이터에 없는 조합은 out-of-distribution 후보로 표시한다.

탐색 방식:

- exhaustive search, 후보 수가 작을 때
- random search
- Bayesian optimization
- rule-based filtering

### 5.3 Ranker

역할:

- 예측 성능, 비용, 공급 안정성, 문헌 근거를 결합해 후보를 정렬한다.

초기 rank score:

```text
score = predicted_capacity_rank
```

후속 rank score:

```text
score = w1 * capacity_score - w2 * cost_score + w3 * supply_score + w4 * evidence_score
```

가중치는 실험 전 고정하지 않고 Phase 6에서 별도 검증한다.

## 6. 안전 및 품질 규칙

- 에이전트 추천은 실험적으로 검증된 사실로 표현하지 않는다.
- 모든 추천에는 "예측 기반 후보"와 "실험 검증 필요" 문구를 포함한다.
- 학습 데이터 분포 밖 조합은 명확히 경고한다.
- 모델의 group split 성능이 낮은 경우 추천 신뢰도를 낮춘다.
- XAI contribution은 모델 예측 해석이며 실제 화학적 인과로 단정하지 않는다.
- 가격/공급망 정보는 출처와 조회 시점을 기록한다.
- 문헌 검색 결과는 예측 결과를 보조하는 근거이지, 실험 검증을 대체하지 않는다.

## 7. 출력 리포트 형식

권장 출력:

```text
1. 요청 목표 요약
2. 사용한 모델과 검증 성능
3. 탐색 공간과 제약 조건
4. 추천 후보 Top 3~10
5. 각 후보의 예측 remain_capacity
6. 주요 feature contribution, 구현 가능한 경우
7. confidence score 및 out-of-distribution 경고
8. 비용/공급망/문헌 근거, 사용 가능한 경우
9. 실험 검증 우선순위
10. 한계와 추가 실험 제안
```

## 8. 구현 시점

본 에이전트는 ML 실험 Phase 1~5 이후 구현한다.

선행 조건:

- best model artifact 존재
- feature schema 고정
- preprocessing pipeline 저장 가능
- random/group split 성능 기록
- prediction API 또는 local inference wrapper 구현

## 9. 비범위

초기 에이전트에서는 다음을 수행하지 않는다.

- 실제 실험 결과라고 단정
- 학습 데이터에 없는 화학식을 임의 생성해 검증 없이 추천
- 외부 가격 정보를 출처 없이 사용
- 특허 회피 또는 안전성 판단을 법적/실험적 결론으로 표현
- ML 모델 재학습
