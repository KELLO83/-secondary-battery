# Dataset Explanation

The active project is dataset-agnostic.

Any CSV can be used when it defines:

- feature columns
- one numeric regression target
- columns to exclude because they are identifiers, metadata, or leakage

## Current Local Sample

AI Hub experimental materials property data:

```text
Experimental materials property data/
  01.원천데이터/
  02.라벨링데이터/
    Resistivity_data.csv
    Hardness_data.csv
    Amorphous_data.csv
```

Recommended regression tasks:

- `Resistivity_data.csv` -> target `Resistivity`
- `Hardness_data.csv` -> target `Hardness`

`Amorphous_data.csv` is a classification-style 0/1 target and is not the primary regression task.

## Resistivity Features

Use:

```text
Al, Ti, Cr, Fe, Co, Ni, Cu, Zr, Mo, W, Mn, Si, Mg,
Resistance, Thickness,
ravg, delta, dHmix, ENavg, dEN, N
```

Exclude:

```text
Number, X, Y, Ex_resistivity, Compo
```

## Hardness Features

Use:

```text
Al, Ti, Cr, Fe, Co, Ni, Cu, Zr, Mo, W, Mn, Si, Mg, Re, Ta,
Thickness,
ravg, delta, dHmix, ENavg, dEN, N
```

Exclude:

```text
Number, X, Y, Compo, Modulus
```
