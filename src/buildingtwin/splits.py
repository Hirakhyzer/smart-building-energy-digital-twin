"""Leakage-resistant chronological train, validation, and test splitting."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import pandas as pd


def _check_fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite fraction strictly between zero and one.")


def _timestamps(frame: pd.DataFrame, time_column: str) -> pd.Series:
    if time_column not in frame.columns:
        raise ValueError(f"Missing time column: {time_column}")
    try:
        values = pd.to_datetime(frame[time_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{time_column} must contain valid timestamps.") from exc
    if values.isna().any():
        raise ValueError(f"{time_column} contains missing timestamps.")
    return values


def chronological_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    time_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by distinct ordered timestamps; never split identical times across sets."""
    _check_fraction(validation_fraction, "validation_fraction")
    _check_fraction(test_fraction, "test_fraction")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Fractions must leave a training period.")

    ordered = frame.copy()
    ordered[time_column] = _timestamps(ordered, time_column)
    ordered = ordered.sort_values(time_column).reset_index(drop=True)
    times = pd.Index(ordered[time_column].drop_duplicates().sort_values())
    if len(times) < 6:
        raise ValueError("Need at least six distinct timestamps for chronological splitting.")

    train_end = times[int(len(times) * (1 - validation_fraction - test_fraction)) - 1]
    validation_end = times[int(len(times) * (1 - test_fraction)) - 1]
    train = ordered.loc[ordered[time_column] <= train_end].copy()
    validation = ordered.loc[
        (ordered[time_column] > train_end) & (ordered[time_column] <= validation_end)
    ].copy()
    test = ordered.loc[ordered[time_column] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split created an empty partition.")
    return train, validation, test


def assert_temporal_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    time_column: str = "timestamp",
) -> None:
    """Reject missing timestamps, empty partitions, and time-window overlap."""
    if any(frame.empty for frame in (train, validation, test)):
        raise AssertionError("All chronological partitions must be nonempty.")
    try:
        start = _timestamps(train, time_column)
        middle = _timestamps(validation, time_column)
        end = _timestamps(test, time_column)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    if start.max() >= middle.min():
        raise AssertionError("Training and validation periods overlap or are out of order.")
    if middle.max() >= end.min():
        raise AssertionError("Validation and test periods overlap or are out of order.")
