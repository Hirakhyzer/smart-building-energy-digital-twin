# Methodology

## Scope

This repository is a research prototype for smart-building energy digital twins. It combines real building meter/weather/metadata inputs with an explicit physics-informed thermal model and transparent scenario studies. It is not a deployed building-management system, a certified energy audit, or a predictive-maintenance product.

## 1. Data foundation

The primary study uses local ASHRAE Great Energy Predictor III tables: meter readings, weather, and building metadata. The code validates required join and time columns, reports missingness and duplicates, and keeps downloaded data out of Git.

The first experiment should use a small, explicitly configured building subset. This avoids hidden sampling decisions and keeps iteration computationally feasible.

## 2. Chronological evaluation

Rows are split by ordered timestamps into training, validation, and test periods. Future observations cannot enter the training period. The model is developed using training/validation data, and the test period is reserved for final evaluation.

A chronological split is necessary for forecasting but does not by itself prove transportability to another site or building class. Cross-building and cross-site generalization are optional later studies.

## 3. Forecasting layer

The initial baseline is HistGradientBoostingRegressor using causal calendar features, available weather inputs, and available static metadata. It provides a robust reference before adding a neural sequence model.

Reported forecast metrics are generated only by executed experiments:

- MAE: average absolute prediction error;
- RMSE: sensitivity to larger errors;
- sMAPE: scale-normalized comparison, interpreted cautiously near zero readings.

The first baseline deliberately avoids target lags because their cadence, missingness, and causal availability must be inspected before use.

## 4. Physics-informed RC digital twin

The building envelope is represented by a two-state resistance-capacitance model with indoor-air temperature \(T_a\) and thermal-mass temperature \(T_m\):

\[
C_a \frac{dT_a}{dt} = \frac{T_o - T_a}{R_{ao}} + \frac{T_m - T_a}{R_{am}} + \eta P_{HVAC} + Q_{internal} + Q_{solar}
\]

\[
C_m \frac{dT_m}{dt} = \frac{T_a - T_m}{R_{am}}
\]

The parameters are scenario/calibration parameters, not universal facts about a building. Positive HVAC power adds heat; cooling is represented by negative power. MATLAB implements the same step equation for cross-environment system modelling.

## 5. Comfort and cost scenarios

A finite, auditable setpoint grid is evaluated under the RC model. For each candidate, the code reports energy proxy, price-weighted cost proxy, comfort violation degree-hours, and a documented composite objective. The built-in price signal is explicitly synthetic unless a validated tariff time series is added.

The scenario output is a research recommendation, not automatic HVAC actuation.

## 6. Degradation scenarios and anomaly analysis

The default energy data do not establish component-level degradation labels. The project therefore does not claim real HVAC fault detection. Instead, it creates transparent scenario injections:

- gradual efficiency decline;
- optional sensor bias after a chosen onset;
- residual-based maintenance risk signal;
- residual-feature Isolation Forest anomaly scoring.

All figures and tables from this module must be titled **simulated scenario**.

## 7. MATLAB workflow

MATLAB reads local Python exports for independent plotting and system-model inspection. The included `rc_thermal_step.m` mirrors the Python RC update. MATLAB figures are only generated after the Python result CSV exists.

## 8. Reproducibility and limitations

- Fixed random seeds are declared in configuration.
- Raw/processed data and generated results are excluded from Git.
- Tests cover data-free thermal dynamics, degradation profile behavior, and chronological split integrity.
- The project needs actual local data inspection before any data-specific metrics, calibrated parameters, or results can be reported.
- Model performance depends on meter quality, weather alignment, building coverage, missingness, and the selected meter type.
