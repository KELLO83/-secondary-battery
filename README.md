# Prediction ML

Generic CSV-based tabular regression model comparison.

Use config-driven runs:

```powershell
.\.venv314\Scripts\python.exe train.py --config configs\nasa_capacity_lightgbm.json
```

Or direct CLI:

```powershell
.\.venv314\Scripts\python.exe train.py --csv data\processed\nasa_cycle_level.csv --target capacity --features sample_count,duration_sec,cycle_index --model lightgbm
```
