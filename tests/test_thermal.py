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
