# Run Risk Checklist

Before running:

- [ ] CSV path exists
- [ ] target column is numeric or convertible to numeric
- [ ] feature columns are known or `exclude` is explicit
- [ ] target-derived columns are excluded
- [ ] row count is large enough for the intended split
- [ ] split strategy matches the dataset structure
- [ ] model environment is correct
- [ ] TabPFN token/checkpoint access is available if using TabPFN

After running:

- [ ] result row exists in `results/tabular_regression_experiments.csv`
- [ ] feature count matches expectation
- [ ] train/validation row counts are reasonable
- [ ] metrics are not dominated by near-zero targets
- [ ] no parallel process corrupted the result CSV
