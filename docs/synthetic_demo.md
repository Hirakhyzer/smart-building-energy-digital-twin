# Synthetic demo mode

## Purpose

`python scripts/run_synthetic_demo.py` makes the repository runnable before any external data download. It generates a deterministic multi-building campus with hourly weather, calendar-driven occupancy proxies, building metadata, meter readings, rare energy spikes, a forecast benchmark, residual anomaly scores, thermal scenarios, and an injected efficiency-decline study.

## What the demo proves

The demo exercises the code paths, output contract, figures, chronology safeguards, and research workflow.

It does **not** demonstrate forecast accuracy on a real building, calibrated RC parameters, a utility tariff, real occupancy, real indoor temperature, or observed HVAC degradation.

## Command

```bash
python scripts/run_synthetic_demo.py
```

Optional controls:

```bash
python scripts/run_synthetic_demo.py --days 120 --buildings 6 --seed 42
```

## Local outputs

```text
outputs/results/synthetic_demo_summary.json
outputs/results/synthetic_demo_predictions.csv
outputs/results/synthetic_demo_anomaly_scores.csv
outputs/results/synthetic_demo_thermal_scenarios.csv
outputs/results/synthetic_demo_degradation.csv

outputs/figures/synthetic_energy_weather_profile.png
outputs/figures/synthetic_forecast_window.png
outputs/figures/synthetic_anomaly_scores.png
outputs/figures/synthetic_comfort_cost_tradeoff.png
outputs/figures/synthetic_degradation_profile.png
```

The synthetic input tables are written under `data/interim/synthetic_ashrae/`, which is ignored by Git.

## Interpretation rules

- Every demo artifact must retain the word **synthetic** in its filename or caption.
- Do not put demo metrics in a PhD application, paper, or report as empirical performance evidence.
- Use the ASHRAE workflow only after downloading and inspecting the real CSV files.
