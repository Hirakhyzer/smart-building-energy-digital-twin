import pandas as pd

from buildingtwin.splits import assert_temporal_order, chronological_split


def test_chronological_split_keeps_future_out_of_training():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=20, freq="h"), "meter_reading": range(20)})
    train, validation, test = chronological_split(frame, validation_fraction=0.2, test_fraction=0.2)
    assert_temporal_order(train, validation, test)
    assert len(train) + len(validation) + len(test) == len(frame)


import numpy as np
import pytest


@pytest.mark.parametrize("fractions", [
    (float("nan"), 0.2),
    (0.2, float("inf")),
    (True, 0.2),
    (0.5, 0.5),
    (0.01, 0.01),
])
def test_split_rejects_invalid_fractions_or_empty_partitions(fractions):
    data = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="h")})
    with pytest.raises(ValueError):
        chronological_split(data, *fractions)


def test_split_rejects_missing_or_invalid_timestamps():
    data = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="h")})
    data.loc[5, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="missing timestamps"):
        chronological_split(data)
    bad = pd.DataFrame({"timestamp": ["not-a-time"] * 10})
    with pytest.raises(ValueError, match="valid timestamps"):
        chronological_split(bad)


def test_split_does_not_distribute_equal_timestamp_across_partitions():
    repeated = np.repeat(pd.date_range("2026-01-01", periods=10, freq="h"), 3)
    frame = pd.DataFrame({"timestamp": repeated, "meter_reading": range(30)})
    shuffled = frame.sample(frac=1, random_state=11).reset_index(drop=True)
    train, valid, test = chronological_split(shuffled, 0.2, 0.2)
    assert_temporal_order(train, valid, test)
    assert set(train.timestamp).isdisjoint(valid.timestamp)
    assert set(valid.timestamp).isdisjoint(test.timestamp)
    assert len(train) + len(valid) + len(test) == len(frame)


def test_split_parses_dates_before_ordering():
    dates = [f"2026-01-{day}" for day in range(1, 11)]
    frame = pd.DataFrame({"timestamp": dates[::-1]})
    train, valid, test = chronological_split(frame, 0.2, 0.2)
    assert_temporal_order(train, valid, test)
    assert train.timestamp.min() == pd.Timestamp("2026-01-01")


def test_temporal_order_rejects_empty_or_missing_time_partitions():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="h")})
    train, valid, test = chronological_split(frame, 0.2, 0.2)
    with pytest.raises(AssertionError, match="nonempty"):
        assert_temporal_order(train.iloc[0:0], valid, test)
    invalid = test.copy()
    invalid.loc[invalid.index[0], "timestamp"] = pd.NaT
    with pytest.raises(AssertionError, match="missing timestamps"):
        assert_temporal_order(train, valid, invalid)
