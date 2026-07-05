"""Counterfactual comfort and energy scenario evaluation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .thermal import RCParameters, hvac_power_for_setpoint, rc_step

@dataclass(frozen=True)
class ScenarioScore:
    target_temperature: float
    energy_proxy: float
    cost_proxy: float
    comfort_degree_hours: float
    objective: float

def score_temperature_target(outdoor_temperature, electricity_price, target_temperature, comfort_min, comfort_max, parameters=None):
    """Evaluate one fixed indoor target with the transparent RC model."""
    outdoor = np.asarray(outdoor_temperature, dtype=float)
    price = np.broadcast_to(np.asarray(electricity_price, dtype=float), outdoor.shape)
    params = parameters or RCParameters()
    air = 22.0; mass = 22.0
    powers = []; temperatures = []
    for outside in outdoor:
        power = hvac_power_for_setpoint(air, target_temperature)
        air, mass = rc_step(air, mass, float(outside), power, 0.0, 0.0, params)
        powers.append(power); temperatures.append(air)
    powers = np.asarray(powers); temperatures = np.asarray(temperatures)
    violation = np.maximum(comfort_min - temperatures, 0) + np.maximum(temperatures - comfort_max, 0)
    energy = float(np.sum(np.abs(powers)))
    cost = float(np.sum(np.abs(powers) * price))
    comfort = float(np.sum(violation))
    return ScenarioScore(float(target_temperature), energy, cost, comfort, cost + 10.0 * comfort)

def choose_best_target(outdoor_temperature, electricity_price, candidates, comfort_min, comfort_max, parameters=None):
    """Return the best entry from an auditable finite grid of target temperatures."""
    scores = [score_temperature_target(outdoor_temperature, electricity_price, value, comfort_min, comfort_max, parameters) for value in candidates]
    if not scores:
        raise ValueError("At least one scenario target is required.")
    return min(scores, key=lambda item: item.objective), scores
