import pandas as pd

from buildingtwin.splits import assert_temporal_order, chronological_split


def test_chronological_split_keeps_future_out_of_training():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=20, freq="h"), "meter_reading": range(20)})
    train, validation, test = chronological_split(frame, validation_fraction=0.2, test_fraction=0.2)
    assert_temporal_order(train, validation, test)
    assert len(train) + len(validation) + len(test) == len(frame)
