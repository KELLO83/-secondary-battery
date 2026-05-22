# Run Risk Checklist

## Data

- [ ] `data/nasa_battery_raw/cleaned_dataset/metadata.csv`가 존재하는가?
- [ ] `data/nasa_battery_raw/cleaned_dataset/data/*.csv`가 존재하는가?
- [ ] `data/processed/nasa_cycle_level.csv`가 최신 원본에서 생성되었는가?
- [ ] discharge row만 1차 모델에 사용했는가?

## Split

- [ ] train/validation이 `battery_id` group holdout인가?
- [ ] 같은 `battery_id`가 train과 validation에 동시에 들어가지 않는가?
- [ ] random row split을 사용하지 않았는가?

## Feature

- [ ] 정상 실험 feature set이 `cycle_basic` 또는 `discharge_summary`인가?
- [ ] `discharge_health` 결과를 leaderboard로 사용하지 않았는가?
- [ ] `soh`가 운영 feature에 들어가지 않았는가?

## Execution

- [ ] 한 번의 `train.py` 실행이 하나의 모델만 학습하는가?
- [ ] 로그 파일이 `results/logs/`에 남는가?
- [ ] `results/experiments.csv`에 metric이 저장되는가?
- [ ] 장시간 GPU 실험 전 LightGBM baseline을 먼저 확인했는가?

## Model-Specific

- [ ] CatBoost 정식 실험은 GPU 기본값을 사용하는가?
- [ ] Transformer/Neural 모델은 과적합 여부를 확인하는가?
- [ ] Foundation model은 pretrained checkpoint/access 조건을 로그에 남기는가?
