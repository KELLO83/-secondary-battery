# Model Tiers

## Tier 0: Sanity Baselines

- Dummy mean
- Ridge

목적:

- split과 metric이 정상인지 확인한다.
- `cycle_basic` 대비 `discharge_summary`의 정보 이득을 확인한다.

## Tier 1: Core GBDT Baselines

1. LightGBM
2. CatBoost

정책:

- LightGBM을 첫 기준 모델로 실행한다.
- CatBoost는 정식 실험에서 GPU를 기본값으로 사용한다.
- 두 모델 모두 `discharge_summary`를 기본 feature set으로 사용한다.

## Tier 2: Neural Tabular Models

1. RealMLP
2. TabM
3. NODE
4. TabR
5. DCN-V2

정책:

- `.venv314`와 CUDA torch 환경을 사용한다.
- cycle-level row 수가 약 2.8K로 작으므로 과적합을 강하게 감시한다.
- GPU batch size는 OOM 없이 가능한 범위에서 크게 잡되, 실험 1회당 모델 1개 원칙을 유지한다.

## Tier 3: Transformer Models

1. FT-Transformer
2. TabTransformer
3. TabNet

주의:

- NASA cycle-level table은 row 수가 크지 않다.
- Transformer는 성능 확인용으로만 우선 실행하고, GBDT/TabM/NODE보다 우선순위를 낮춘다.

## Tier 4: Foundation / Training-Free Models

1. TabICLv2
2. TabPFN 계열

정책:

- 가능한 경우 pretrained checkpoint를 사용한다.
- TabPFN 계열은 token/checkpoint/license 조건을 실험 로그에 남긴다.
- fine-tuning이 아니라 in-context inference 방식인지 명확히 기록한다.

## Tier 5: Ceiling Benchmark

- AutoGluon/Mitra

정책:

- 설치/라이선스/실행 비용을 확인한 뒤 별도 ceiling benchmark로만 실행한다.
- 기본 실험 자동화에는 포함하지 않는다.

## Execution Rule

```text
one train.py invocation = one model experiment
```

여러 모델 sweep을 자동으로 이어서 실행하지 않는다.
