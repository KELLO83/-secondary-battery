# RUN_RISK_CHECKLIST: 대형 모델 실험 실행 전 리스크 점검

## 1. 목적

본 문서는 `remain_capacity` 모델 실험을 실행하기 전에 반드시 확인해야 하는 운영 리스크를 정리한다.

특히 다음 3가지는 실험 실패 또는 결과 해석 오류를 만들 수 있으므로 사전 점검한다.

- GPU memory/OOM
- TabPFN token/checkpoint/license
- AutoGluon/Mitra 설치 및 실행 비용

## 2. GPU Memory / OOM

### 대상 모델

```text
CatBoost GPU
RealMLP
TabM
TabR
DCN-V2
NODE
FT-Transformer
TabTransformer
TabNet
TabPFN latest
TabICLv2
AutoGluon/Mitra
```

### 기본 정책

- `.venv314`에서 실행한다.
- GPU 사용 가능 모델은 GPU를 우선 사용한다.
- 목표 GPU memory 사용률은 최대 90%다.
- 장시간 대형 실험은 한 번에 하나의 모델만 실행한다.
- sweep/grid/연속 leaderboard 자동 실행은 금지한다.

### 실행 전 확인

```powershell
nvidia-smi
```

확인 항목:

- 다른 프로세스가 GPU memory를 점유하고 있는지
- 사용 가능한 VRAM이 충분한지
- 노트북 RTX 4060 8GB 환경에서 1M 이상 neural/transformer 실험을 바로 시작하지 않는지

### sample size 확장 순서

```text
50k -> 100k -> 500k -> 1M -> full, 가능한 모델만
```

모델별 기본 상한:

| 모델군 | 권장 시작 | 확장 후보 | 주의 |
| :--- | ---: | ---: | :--- |
| LightGBM | 100k | full | CPU 중심, 가장 먼저 full 검증 |
| CatBoost GPU | 100k | full | `task_type=GPU`, `gpu_ram_part=0.90` |
| RealMLP/TabM | 100k | 1M~full | batch size 조정 필수 |
| TabR | 50k | 500k~1M | retrieval/index memory 주의 |
| DCN-V2/NODE | 100k | 500k~1M | OOM 시 depth/layers/batch 축소 |
| FT/TabTransformer | 50k | 500k~1M | attention 비용 주의 |
| TabNet | 50k | 500k | 보조 실험 |
| TabPFN-3 latest | 100k | 500k~1M | token/checkpoint와 VRAM 조건 필수 |
| TabICLv2 | 50k | 500k~1M | checkpoint/kv cache memory 확인 |

### OOM 발생 시 대응 순서

1. `batch_size` 축소
2. `eval_batch_size` 또는 prediction chunk 축소
3. embedding dimension 축소
4. model depth/layers/trees 축소
5. sample size 축소
6. CPU fallback 여부를 명시적으로 기록

### 기록 필수 항목

```text
device
gpu_name
gpu_memory_total
gpu_memory_used_peak
batch_size
eval_batch_size
sample_size
oom 여부
fallback 여부
```

## 3. TabPFN Token / Checkpoint / License

### 대상 모델

```text
tabpfn
tabpfn_latest
```

### 현재 상태

TabPFN 계열 wrapper는 token 또는 local checkpoint가 없으면 의도적으로 실패한다.

정식 실험은 브라우저 로그인 팝업에 의존하지 않는다.

허용되는 접근 방식:

```text
TABPFN_TOKEN 환경변수
~/.cache/tabpfn/auth_token
~/.tabpfn/token
명시적 local model_path/checkpoint
```

### 실행 전 확인

```powershell
$env:TABPFN_TOKEN
Test-Path $HOME\.cache\tabpfn\auth_token
Test-Path $HOME\.tabpfn\token
```

### 실험 정책

- 기본은 pretrained inference / in-context 실험이다.
- from-scratch 학습으로 대체하지 않는다.
- token/checkpoint가 없으면 실패 실험으로 기록한다.
- TabPFN-3 local/OSS 경로는 전체 16.3M rows 기본 실험으로 보지 않는다.
- `core_11` 기준 100k, 500k, 가능하면 1M sample까지 검토한다.
- API/Enterprise Scaling Mode는 별도 large-scale foundation 실험으로 분리한다.

### 기록 필수 항목

```text
tabpfn_version
checkpoint
weight_source
access_mode
license_checked
token_present
local_checkpoint_path
api_or_local
```

## 4. AutoGluon/Mitra 설치 및 실행 조건

### 대상 모델

```text
autogluon_mitra
```

### 현재 상태

AutoGluon/Mitra는 optional ceiling benchmark이며, 기본 단일 모델 leaderboard와 섞지 않는다.

현재 wrapper는 `autogluon.tabular`가 설치되어 있지 않으면 의도적으로 실패한다.

### 설치 원칙

`.venv314`에서만 설치를 검토한다.

```powershell
.\.venv314\Scripts\python.exe -m pip install "autogluon.tabular[mitra]"
```

주의:

- `.venv314t`에 설치하지 않는다.
- 소스 빌드로 억지 설치하지 않는다.
- 설치 후 PyTorch/CUDA/pandas/scikit-learn dependency 충돌을 확인한다.

### 실행 정책

- ceiling benchmark로만 사용한다.
- 단일 모델 순위와 섞지 않는다.
- 먼저 50k 또는 100k sample로 설치/fit/predict smoke를 확인한다.
- `time_limit`을 반드시 지정한다.
- full data는 Tier 1~3 단일 모델 결과를 본 뒤에만 검토한다.

### 기록 필수 항목

```text
autogluon_version
mitra_enabled
time_limit
presets
hyperparameters
included_model_types
excluded_model_types
train_time_sec
predict_time_sec
artifact_size
```

## 5. 실험 시작 전 최종 체크

```text
[ ] feature_set이 core_11/design_15/chem_22/chem_derived 중 하나인가?
[ ] official/raw-full feature set을 사용하지 않는가?
[ ] discharge_capacity/state_of_charge가 feature에 없는가?
[ ] sample 기반 실험이면 source_family 비율 유지 sampling인가?
[ ] GPU 사용 모델이면 nvidia-smi 확인을 했는가?
[ ] 대형 실험이면 한 번에 하나의 모델만 실행하는가?
[ ] TabPFN이면 token/checkpoint/license를 확인했는가?
[ ] AutoGluon/Mitra이면 설치 여부와 time_limit을 확인했는가?
[ ] 실패 실험도 results에 기록할 준비가 되어 있는가?
```
