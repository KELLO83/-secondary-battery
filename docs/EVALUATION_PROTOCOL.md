# EVALUATION_PROTOCOL: 모델 평가 및 비교 규칙

## 1. 목적

본 문서는 2차전지 `remain_capacity` 예측 모델을 공정하게 비교하기 위한 평가 규칙을 정의한다.

모델 수가 많기 때문에, 성능 비교는 동일한 데이터 조건, 동일한 split, 동일한 metric에서만 수행한다.

## 2. 평가 Metric

필수 metric:

```text
MAPE
MAE
RMSE
```

부가 기록:

```text
train_time_sec
predict_time_sec
peak_memory_mb
gpu_memory_mb
```

주의:

- MAPE는 target이 0에 가까운 경우 불안정하다.
- MAPE 단독으로 최종 모델을 결정하지 않는다.
- MAE, RMSE, group split 성능을 함께 본다.

## 3. Split 규칙

### 3.1 Random Split

목적:

- 일반 benchmark 성능 확인
- 기존 benchmark와 비교 가능한 기준 확보

기본 비율:

```text
train: 80%
valid: 10%
test: 10%
```

### 3.2 Group Split

목적:

- 논문/소재 단위 leakage 가능성 검증
- 신규 문헌 또는 신규 소재 예측 상황에 가까운 성능 확인

group key 우선순위:

```text
1. DOI
2. material_id
3. chemical_formula
```

주의:

- group key는 feature로 사용하지 않는다.
- train/valid/test 간 group overlap이 있으면 해당 split은 무효다.

## 4. 비교 규칙

동일 조건에서만 직접 비교한다.

동일 조건의 기준:

```text
same sample_size
same split_type
same split_seed
same group_key
same feature_set
same target_filter
same preprocessing_version
same training_mode, 직접 비교 시
```

금지:

- 100k sample 모델과 full-data 모델을 직접 우열 비교
- random split 결과와 group split 결과를 같은 leaderboard에서 순위 비교
- pretrained foundation model 결과와 from-scratch 결과를 구분 없이 비교
- training-free/in-context foundation model과 gradient 학습 모델의 학습 시간을 같은 의미로 비교
- metric 계산 방식이 다른 결과를 같은 표에 혼합
- 48개 raw column 전체를 그대로 feature로 사용하는 것

## 4.1 통합 모델 및 계열별 분석 규칙

본 프로젝트의 기본 평가는 LFP, NCA, NCM, Others를 병합한 통합 모델 기준으로 수행한다.

훈련:

```text
Training/LFP_train_dataset.csv
Training/NCA_train_dataset.csv
Training/NCM_train_dataset.csv
Training/others_train_dataset.csv
-> 하나의 train dataframe으로 병합
-> 통합 모델 1개 학습
```

검증:

```text
Validation/LFP_val_dataset.csv
Validation/NCA_val_dataset.csv
Validation/NCM_val_dataset.csv
Validation/others_val_dataset.csv
-> 하나의 validation dataframe으로 병합
-> 전체 metric 계산
-> source_family별 metric 별도 계산
```

규칙:

- `source_family`는 학습 feature가 아니다.
- `source_family`별 MAPE/MAE/RMSE는 분석용 metric으로 기록한다.
- 1차 기본 평가는 `core_11` feature set 기준으로 수행한다.
- 먼저 `core_11`에서 우수한 모델군을 찾고, `design_15`, `chem_22`, `official`은 후속 모델로 분리 평가한다.
- `design_15`는 `core_11`에 `sintering_T1(C)`, `sintering_t1(h)`, `measurement_T(C)`, `C-rate`를 추가한 모델이다.
- `chem_22`는 `design_15`에 `Li_fraction`, `Ni_fraction`, `Mn_fraction`, `Co_fraction`, `dopant_fraction`, `active_proportion`, `binder_proportion`을 추가한 모델이다.
- 계열별 specialized model은 기본 실험 범위에서 제외한다.
- 특정 계열에서 통합 모델 성능이 지속적으로 낮을 때만 Phase 5 이후 후속 실험으로 검토한다.

## 5. Leaderboard 규칙

leaderboard는 split type별로 분리한다.

```text
results/leaderboard_random.csv
results/leaderboard_group.csv
```

정렬 기준:

```text
1. test_mape ascending
2. test_mae ascending
3. test_rmse ascending
4. predict_time_sec ascending
5. train_time_sec ascending
```

## 6. 최종 모델 선정 기준

최종 운영 후보는 단순 test MAPE 1등이 아니라 다음 기준으로 고른다.

우선순위:

```text
1. group split test MAPE
2. group split test MAE/RMSE
3. random split과 group split의 성능 차이
4. 추론 시간
5. artifact 저장/로드 안정성
6. OOD 경고 및 XAI 적용 가능성
7. 라이선스 및 배포 가능성
```

운영 후보와 연구 후보를 분리한다.

운영 후보:

```text
LightGBM
CatBoost
RealMLP
TabM
DCN-V2
```

연구 후보:

```text
FT-Transformer
TabTransformer
TabPFN-3
TabICLv2
TabNet
```

## 7. 실패 실험 기록

실패한 실험도 기록한다.

기록 항목:

```text
experiment_id
model_name
sample_size
split_type
failure_stage
error_message
oom
elapsed_time_sec
notes
```

목적:

- 반복 OOM 방지
- 실행 불가능한 설정 추적
- 모델별 실용성 판단

## 8. 보고 규칙

최종 리포트에는 다음을 포함한다.

- random split leaderboard
- group split leaderboard
- 모델별 성능/시간/메모리 비교
- GBDT vs neural vs foundation model 비교
- XAI/OOD 적용 가능성
- 최종 운영 후보
- 연구 후보
- 한계와 후속 실험
