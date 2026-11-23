"""Regression tests for comfort-cost scenario input integrity."""

import numpy as np
import pytest

from buildingtwin.scenarios import choose_best_target, score_temperature_target


def test_best_target_matches_minimum_auditable_objective():
    weather = [10.0, 11.0, 13.0, 12.0]
    best, scores = choose_best_target(weather, 1.0, [20.0, 21.0, 22.0], 19.0, 24.0)
    assert len(scores) == 3
    assert best.objective == min(score.objective for score in scores)
    assert all(np.isfinite(score.objective) for score in scores)


@pytest.mark.parametrize(("name", "value"), [
    ("outdoor_temperature", []),
    ("outdoor_temperature", [float("nan")]),
    ("outdoor_temperature", [[10.0, 11.0]]),
    ("electricity_price", -1.0),
    ("electricity_price", float("inf")),
    ("electricity_price", [1.0, 1.0, 1.0]),
    ("target_temperature", float("nan")),
    ("comfort_min", float("-inf")),
    ("comfort_max", float("nan")),
])
def test_invalid_scenario_values_rejected(name, value):
    values = dict(
        outdoor_temperature=[10.0, 12.0],
        electricity_price=[1.0, 2.0],
        target_temperature=22.0,
        comfort_min=19.0,
        comfort_max=25.0,
    )
    values[name] = value
    with pytest.raises(ValueError):
        score_temperature_target(**values)


def test_invalid_comfort_interval_rejected():
    with pytest.raises(ValueError, match="comfort_min"):
        score_temperature_target([10.0], [1.0], 22.0, 26.0, 19.0)


def test_empty_or_nonfinite_candidate_grid_rejected():
    with pytest.raises(ValueError, match="At least one"):
        choose_best_target([10.0], [1.0], [], 19.0, 25.0)
    with pytest.raises(ValueError, match="target_temperature"):
        choose_best_target([10.0], [1.0], [20.0, float("nan")], 19.0, 25.0)
