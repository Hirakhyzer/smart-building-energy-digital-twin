"""ASHRAE-style local data loading, validation, and inspection.

The loader checks documented expected column names and fails clearly when a local
copy differs. No downloaded data are committed or fabricated by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIN_REQUIRED = {"building_id", "meter", "timestamp", "meter_reading"}
WEATHER_REQUIRED = {"site_id", "timestamp"}
METADATA_REQUIRED = {"site_id", "building_id"}


def load_ashrae_tables(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load local ASHRAE input tables and validate minimum join/time columns."""
    root = Path(raw_dir)
    paths = {
        "train": root / "train.csv",
        "weather": root / "weather_train.csv",
        "metadata": root / "building_metadata.csv",
    }
    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing local ASHRAE files: " + ", ".join(missing_files))
    train = pd.read_csv(paths["train"])
    weather = pd.read_csv(paths["weather"])
    metadata = pd.read_csv(paths["metadata"])
    _require_columns(train, TRAIN_REQUIRED, "train.csv")
    _require_columns(weather, WEATHER_REQUIRED, "weather_train.csv")
    _require_columns(metadata, METADATA_REQUIRED, "building_metadata.csv")
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="raise")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="raise")
    return train, weather, metadata


def inspect_ashrae_tables(raw_dir: str | Path) -> dict[str, Any]:
    """Return machine-readable source, time, meter, and missingness summaries."""
    train, weather, metadata = load_ashrae_tables(raw_dir)
    return {
        "train": _frame_summary(train),
        "weather": _frame_summary(weather),
        "metadata": _frame_summary(metadata),
        "train_time_range": _time_range(train),
        "weather_time_range": _time_range(weather),
        "meter_counts": {str(key): int(value) for key, value in train["meter"].value_counts(dropna=False).sort_index().items()},
        "building_count": int(train["building_id"].nunique()),
        "site_count": int(metadata["site_id"].nunique()),
    }


def build_energy_frame(
    train: pd.DataFrame,
    weather: pd.DataFrame,
    metadata: pd.DataFrame,
    meter_type: int | None = 0,
) -> pd.DataFrame:
    """Join meter readings to building metadata and weather with explicit checks.

    `meter_type` is a configuration choice. It defaults to zero for the standard
    ASHRAE electricity meter convention but must be confirmed during inspection.
    """
    readings = train.copy()
    if meter_type is not None:
        readings = readings.loc[readings["meter"] == meter_type].copy()
    if readings.empty:
        raise ValueError("No readings remain after the configured meter_type filter.")
    frame = readings.merge(metadata, on="building_id", how="left", validate="many_to_one")
    if frame["site_id"].isna().any():
        raise ValueError("Metadata join left meter rows without a site_id.")
    frame = frame.merge(weather, on=["site_id", "timestamp"], how="left", suffixes=("", "_weather"), validate="many_to_one")
    frame = frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal calendar features from the observation timestamp."""
    output = frame.copy()
    timestamp = pd.to_datetime(output["timestamp"])
    output["hour"] = timestamp.dt.hour
    output["day_of_week"] = timestamp.dt.dayofweek
    output["month"] = timestamp.dt.month
    output["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    return output


def save_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write inspection summary as readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def _time_range(frame: pd.DataFrame) -> dict[str, str | None]:
    timestamp = pd.to_datetime(frame["timestamp"])
    return {"start": timestamp.min().isoformat() if len(timestamp) else None, "end": timestamp.max().isoformat() if len(timestamp) else None}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")
