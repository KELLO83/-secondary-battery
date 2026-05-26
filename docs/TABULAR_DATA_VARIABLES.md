# Tabular Data Variables

## Variable Types

Numeric:

- continuous measurements
- counts
- engineered scalar features

Categorical:

- labels
- region/site/batch identifiers
- material or product category

Metadata:

- row ID
- file name
- acquisition ID
- coordinates

Target:

- numeric value to predict

## Feature Policy

Use a column as a feature only if it is available at prediction time.

Exclude columns that:

- are the target
- are directly computed from the target
- leak future/post-outcome information
- are pure row identifiers
- should define the split rather than be used as input

## Practical Recommendation

Prefer explicit config files for serious runs:

```json
{
  "features": ["f1", "f2"],
  "exclude": ["id", "target_derived_value"]
}
```

Use CLI `--features` for quick experiments.
