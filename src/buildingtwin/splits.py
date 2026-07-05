"""Chronological train, validation, and test splitting for time-series energy data."""

from __future__ import annotations

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    time_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by globally ordered timestamps to prevent future-data leakage."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must lie between zero and one.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Fractions must leave a training period.")
    ordered = frame.sort_values(time_column).reset_index(drop=True)
    times = pd.Index(ordered[time_column].drop_duplicates().sort_values())
    if len(times) < 6:
        raise ValueError("Need at least six distinct timestamps for chronological splitting.")
    train_end = times[int(len(times) * (1 - validation_fraction - test_fraction)) - 1]
    validation_end = times[int(len(times) * (1 - test_fraction)) - 1]
    train = ordered.loc[ordered[time_column] <= train_end].copy()
    validation = ordered.loc[(ordered[time_column] > train_end) & (ordered[time_column] <= validation_end)].copy()
    test = ordered.loc[ordered[time_column] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split created an empty partition.")
    return train, validation, test


def assert_temporal_order(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, time_column: str = "timestamp") -> None:
    """Raise when any validation/test record precedes training/validation data."""
    if train[time_column].max() >= validation[time_column].min():
        raise AssertionError("Training and validation periods overlap or are out of order.")
    if validation[time_column].max() >= test[time_column].min():
        raise AssertionError("Validation and test periods overlap or are out of order.")
