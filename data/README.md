# Dataset guide and local-data boundary

This repository does **not** include downloaded building-energy data. Keep all raw and derived data local.

## Primary dataset to download

Use the **ASHRAE Great Energy Predictor III** competition data from Kaggle. Download and extract these files locally:

```text
data/raw/ashrae/
├── train.csv
├── weather_train.csv
└── building_metadata.csv
```

The project uses these three tables for a reproducible energy-forecasting and digital-twin calibration workflow. The competition's test files are not required for the first study because this repository creates chronological local train/validation/test partitions from the available historical data.

## What each input is expected to provide

| File | Expected role in this project | Do not assume until inspection |
| --- | --- | --- |
| `train.csv` | Time-stamped meter readings and building/site references | Meter reading units, exact row count, site coverage |
| `weather_train.csv` | Site-level weather time series | Available weather fields and missingness |
| `building_metadata.csv` | Building attributes | Available metadata, building classes, floor-area completeness |

The inspection script verifies actual columns, types, time coverage, missing values, duplicates, meter types, site counts, and plausible join keys. It does not silently coerce an unknown schema.

## Optional future source

The physics/control layer is intentionally dataset-independent. Later, an optional BOPTEST-compatible integration can be added for benchmark control experiments. It is **not** required to start this repository.

## Degradation boundary

ASHRAE meter data do not provide confirmed ground-truth HVAC degradation labels. The first version of this project therefore uses **transparent, model-injected degradation scenarios** for maintenance and control sensitivity studies. Any such output must be labeled as simulated, not as measured HVAC failure detection.

## Data-handling rules

- Do not commit downloaded CSV files, processed subsets, trained models, or result figures unless explicitly permitted.
- Keep data under `data/raw/`, `data/interim/`, and `data/processed/`; all are ignored by Git.
- Upload the three downloaded CSV files to this chat when you are ready for local inspection and integration.
- Retain the original Kaggle license/terms alongside your downloaded data.
