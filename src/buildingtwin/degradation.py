"""Transparent model-injected HVAC degradation and sensor-fault scenarios.

These are simulation studies because the primary meter dataset has no confirmed
component-level maintenance labels. They must never be presented as observed
HVAC fault ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationScenario:
    """Parameterized efficiency-decay and sensor-bias scenario."""

    efficiency_drop_fraction: float = 0.15
    onset_fraction: float = 0.45
    sensor_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.efficiency_drop_fraction < 1:
            raise ValueError("efficiency_drop_fraction must be in [0, 1).")
        if not 0 <= self.onset_fraction < 1:
            raise ValueError("onset_fraction must be in [0, 1).")


def degradation_profile(length: int, scenario: DegradationScenario) -> np.ndarray:
    """Return a gradual efficiency multiplier with value one before onset."""
    if length <= 0:
        raise ValueError("length must be positive.")
    profile = np.ones(length, dtype=float)
    onset = min(length - 1, int(round(length * scenario.onset_fraction)))
    if onset < length - 1:
        decline = np.linspace(0, scenario.efficiency_drop_fraction, length - onset)
        profile[onset:] = 1.0 - decline
    return profile


def inject_degradation_scenario(
    frame: pd.DataFrame,
    energy_column: str = "meter_reading",
    temperature_column: str | None = "air_temperature",
    scenario: DegradationScenario | None = None,
) -> pd.DataFrame:
    """Create a clearly labeled synthetic degradation scenario from a time series.

    Meter use increases inversely with synthetic efficiency. An optional sensor
    bias is applied after onset. Original columns remain unchanged.
    """
    if energy_column not in frame.columns:
        raise KeyError(f"Missing energy column: {energy_column}")
    output = frame.copy().reset_index(drop=True)
    active = scenario or DegradationScenario()
    efficiency = degradation_profile(len(output), active)
    output["simulated_efficiency"] = efficiency
    output["simulated_degradation_active"] = (efficiency < 1.0).astype(int)
    output["simulated_meter_reading"] = output[energy_column].to_numpy(dtype=float) / np.maximum(efficiency, 1e-6)
    if temperature_column is not None and temperature_column in output.columns:
        onset = int(round(len(output) * active.onset_fraction))
        biased = output[temperature_column].to_numpy(dtype=float).copy()
        biased[onset:] += active.sensor_bias
        output[f"simulated_{temperature_column}"] = biased
    return output


def maintenance_risk_index(residual: np.ndarray, smoothing_window: int = 24) -> np.ndarray:
    """Compute a transparent residual-based risk signal for scenario analysis."""
    values = np.asarray(residual, dtype=float)
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least one.")
    absolute = np.abs(values)
    return pd.Series(absolute).rolling(smoothing_window, min_periods=1).mean().to_numpy()
