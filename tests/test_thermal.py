import numpy as np

from buildingtwin.thermal import RCParameters, hvac_power_for_setpoint, rc_step, simulate_rc_model


def test_hvac_heating_increases_next_air_temperature():
    parameters = RCParameters()
    colder, _ = rc_step(20.0, 20.0, 10.0, 0.0, 0.0, 0.0, parameters)
    warmer, _ = rc_step(20.0, 20.0, 10.0, 5.0, 0.0, 0.0, parameters)
    assert warmer > colder


def test_simulation_preserves_horizon_length():
    result = simulate_rc_model(np.array([10.0, 11.0, 12.0]), np.array([1.0, 1.0, 1.0]))
    assert result["air_temperature"].shape == (3,)
    assert result["mass_temperature"].shape == (3,)


def test_proportional_hvac_action_is_bounded():
    assert hvac_power_for_setpoint(0.0, 100.0, max_abs_power=4.0) == 4.0


import pytest


@pytest.mark.parametrize(("name", "value"), [
    ("r_ao", float("nan")),
    ("r_am", float("inf")),
    ("c_air", float("-inf")),
    ("c_mass", 0.0),
    ("hvac_efficiency", True),
])
def test_rc_parameters_reject_nonfinite_or_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        RCParameters(**{name: value})


@pytest.mark.parametrize(("name", "value"), [
    ("dt_hours", float("nan")),
    ("dt_hours", float("inf")),
    ("outdoor_temperature", float("nan")),
    ("hvac_power", float("inf")),
    ("internal_gain", float("-inf")),
])
def test_rc_step_rejects_invalid_inputs(name, value):
    inputs = dict(air_temperature=20.0, mass_temperature=20.0,
                  outdoor_temperature=10.0, hvac_power=1.0,
                  internal_gain=0.0, solar_gain=0.0,
                  parameters=RCParameters(), dt_hours=1.0)
    inputs[name] = value
    with pytest.raises(ValueError, match=name):
        rc_step(**inputs)


@pytest.mark.parametrize(("outdoor", "hvac"), [
    ([], []),
    ([[10.0, 12.0]], [[1.0, 1.0]]),
    ([10.0, 11.0], [1.0]),
    ([float("nan")], [1.0]),
    ([10.0], [float("inf")]),
])
def test_thermal_trajectory_rejects_invalid_horizons(outdoor, hvac):
    with pytest.raises(ValueError):
        simulate_rc_model(np.asarray(outdoor), np.asarray(hvac))


def test_thermal_trajectory_rejects_nonfinite_gain():
    with pytest.raises(ValueError, match="finite"):
        simulate_rc_model(np.array([10.0, 11.0]), np.zeros(2),
                          solar_gain=float("nan"))


@pytest.mark.parametrize(("name", "value"), [
    ("air_temperature", float("nan")),
    ("setpoint", float("inf")),
    ("proportional_gain", 0.0),
    ("max_abs_power", float("inf")),
])
def test_setpoint_controller_rejects_invalid_inputs(name, value):
    args = dict(air_temperature=20.0, setpoint=22.0,
                proportional_gain=1.0, max_abs_power=10.0)
    args[name] = value
    with pytest.raises(ValueError):
        hvac_power_for_setpoint(**args)
