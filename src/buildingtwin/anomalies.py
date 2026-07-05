"""Unsupervised anomaly scoring for meter and sensor scenario analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def build_residual_features(frame: pd.DataFrame, observed_column: str, predicted_column: str) -> pd.DataFrame:
    """Create transparent residual and rolling-residual features."""
    if observed_column not in frame or predicted_column not in frame:
        raise KeyError("Observed and predicted columns must exist.")
    result = pd.DataFrame(index=frame.index)
    result["residual"] = frame[observed_column].to_numpy(dtype=float) - frame[predicted_column].to_numpy(dtype=float)
    result["absolute_residual"] = result["residual"].abs()
    result["rolling_absolute_residual"] = result["absolute_residual"].rolling(24, min_periods=1).mean()
    return result

def fit_anomaly_detector(features: pd.DataFrame, seed: int = 42) -> IsolationForest:
    """Fit an Isolation Forest to numerical residual features."""
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    return model.fit(features)

def score_anomalies(model: IsolationForest, features: pd.DataFrame) -> pd.DataFrame:
    """Return anomaly scores where larger scores indicate greater anomaly evidence."""
    output = features.copy()
    output["anomaly_score"] = -model.score_samples(features)
    return output
