"""Two-state RC thermal model for transparent building digital-twin scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RCParameters:
    """Thermal parameters for a single-zone two-state resistance-capacitance model.

    Units must be chosen consistently by the caller. Parameters are scenario or
    calibration outputs, not universal properties of all buildings.
    """

    r_ao: float = 2.5
    r_am: float = 1.0
    c_air: float = 2.0
    c_mass: float = 12.0
    hvac_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if min(self.r_ao, self.r_am, self.c_air, self.c_mass, self.hvac_efficiency) <= 0:
            raise ValueError("RC resistances, capacitances, and efficiency must be positive.")


def rc_step(
    air_temperature: float,
    mass_temperature: float,
    outdoor_temperature: float,
    hvac_power: float,
    internal_gain: float,
    solar_gain: float,
    parameters: RCParameters,
    dt_hours: float = 1.0,
) -> tuple[float, float]:
    """Advance a two-state RC thermal model by one discrete time step.

    Positive HVAC power adds heat. Cooling can be represented by negative power.
    The simulator is intended for scenario analysis, parameter calibration, and
    controller comparison—not direct equipment actuation.
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive.")
    p = parameters
    air_flow = (outdoor_temperature - air_temperature) / p.r_ao
    mass_flow = (mass_temperature - air_temperature) / p.r_am
    hvac_gain = p.hvac_efficiency * hvac_power
    next_air = air_temperature + dt_hours * (air_flow + mass_flow + hvac_gain + internal_gain + solar_gain) / p.c_air
    next_mass = mass_temperature + dt_hours * (air_temperature - mass_temperature) / (p.r_am * p.c_mass)
    return float(next_air), float(next_mass)


def simulate_rc_model(
    outdoor_temperature: np.ndarray,
    hvac_power: np.ndarray,
    internal_gain: np.ndarray | float = 0.0,
    solar_gain: np.ndarray | float = 0.0,
    initial_air_temperature: float = 22.0,
    initial_mass_temperature: float = 22.0,
    parameters: RCParameters | None = None,
    dt_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Simulate indoor air and mass temperature trajectories over a scenario horizon."""
    outdoor = np.asarray(outdoor_temperature, dtype=float)
    hvac = np.asarray(hvac_power, dtype=float)
    if outdoor.shape != hvac.shape:
        raise ValueError("outdoor_temperature and hvac_power must share shape.")
    internal = np.broadcast_to(np.asarray(internal_gain, dtype=float), outdoor.shape)
    solar = np.broadcast_to(np.asarray(solar_gain, dtype=float), outdoor.shape)
    params = parameters or RCParameters()
    air = np.empty(len(outdoor), dtype=float)
    mass = np.empty(len(outdoor), dtype=float)
    current_air, current_mass = float(initial_air_temperature), float(initial_mass_temperature)
    for index in range(len(outdoor)):
        current_air, current_mass = rc_step(
            current_air, current_mass, outdoor[index], hvac[index], internal[index], solar[index], params, dt_hours
        )
        air[index], mass[index] = current_air, current_mass
    return {"air_temperature": air, "mass_temperature": mass}


def hvac_power_for_setpoint(
    air_temperature: float,
    setpoint: float,
    proportional_gain: float = 1.0,
    max_abs_power: float = 10.0,
) -> float:
    """Return bounded proportional HVAC action for scenario-level control studies."""
    if proportional_gain <= 0 or max_abs_power <= 0:
        raise ValueError("Controller gain and maximum power must be positive.")
    return float(np.clip(proportional_gain * (setpoint - air_temperature), -max_abs_power, max_abs_power))
