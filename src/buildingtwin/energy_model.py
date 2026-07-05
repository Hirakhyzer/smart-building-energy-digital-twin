"""Energy-demand forecasting baseline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

WEATHER_COLUMNS = ("air_temperature", "dew_temperature", "cloud_coverage", "sea_level_pressure", "wind_direction", "wind_speed")
TIME_COLUMNS = ("hour", "day_of_week", "month", "is_weekend")

@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    smape: float

def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select causal calendar, weather, and available static building features."""
    candidates = list(TIME_COLUMNS) + list(WEATHER_COLUMNS) + ["square_feet", "year_built", "floor_count", "meter", "site_id"]
    features = [column for column in candidates if column in frame.columns]
    if not features:
        raise ValueError("No supported forecast features found.")
    x = frame[features].copy()
    for column in x.select_dtypes(include=np.number).columns:
        x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, features

def fit_baseline(x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> HistGradientBoostingRegressor:
    """Fit a computationally safe gradient-boosting baseline."""
    model = HistGradientBoostingRegressor(learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1e-3, random_state=seed)
    return model.fit(x_train, y_train)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    """Compute MAE, RMSE, and symmetric mean absolute percentage error."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    smape = float(np.mean(2 * np.abs(prediction - truth) / np.maximum(np.abs(truth) + np.abs(prediction), 1e-8)))
    return ForecastMetrics(float(mean_absolute_error(truth, prediction)), float(np.sqrt(mean_squared_error(truth, prediction))), smape)
