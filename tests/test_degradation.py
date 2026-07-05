import numpy as np
import pandas as pd

from buildingtwin.degradation import DegradationScenario, degradation_profile, inject_degradation_scenario


def test_degradation_profile_starts_at_one_and_declines():
    profile = degradation_profile(10, DegradationScenario(efficiency_drop_fraction=0.2, onset_fraction=0.5))
    assert np.allclose(profile[:5], 1.0)
    assert profile[-1] < 1.0


def test_scenario_keeps_observed_energy_and_adds_simulated_energy():
    frame = pd.DataFrame({"meter_reading": [10.0, 10.0, 10.0, 10.0]})
    output = inject_degradation_scenario(frame, scenario=DegradationScenario(efficiency_drop_fraction=0.2, onset_fraction=0.5))
    assert "simulated_meter_reading" in output
    assert np.allclose(output["meter_reading"], 10.0)
    assert output["simulated_meter_reading"].iloc[-1] > 10.0
