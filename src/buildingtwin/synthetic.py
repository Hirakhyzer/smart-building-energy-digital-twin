"""Deterministic synthetic smart-building data for an immediate end-to-end demo.

The generated records intentionally resemble the table shapes used by the local
ASHRAE workflow, but they are not ASHRAE records and must never be reported as
real-building measurements. Their only purpose is to make the repository
runnable before downloaded data are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticCampusConfig:
    """Configuration for a reproducible synthetic multi-building campus."""

    days: int = 90
    n_buildings: int = 4
    seed: int = 42
    start: str = "2025-01-01 00:00:00"

    def __post_init__(self) -> None:
        if self.days < 14:
            raise ValueError("Use at least 14 days so chronological splits are meaningful.")
        if self.n_buildings < 1:
            raise ValueError("n_buildings must be positive.")


def generate_synthetic_ashrae_like_data(
    config: SyntheticCampusConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic meter, weather, and building-metadata tables.

    The energy response combines calendar occupancy, weather-sensitive heating and
    cooling demand, building size, meter noise, and rare synthetic energy spikes.
    Values are simulation outputs rather than real physical measurements.
    """
    cfg = config or SyntheticCampusConfig()
    rng = np.random.default_rng(cfg.seed)
    timestamps = pd.date_range(cfg.start, periods=cfg.days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofweek.to_numpy()
    day_index = np.arange(len(timestamps)) / 24.0

    daily_wave = 8.0 * np.sin(2.0 * np.pi * (hour - 7) / 24.0)
    slow_wave = 4.0 * np.sin(2.0 * np.pi * day_index / max(cfg.days, 1))
    outdoor_temperature = 14.0 + daily_wave + slow_wave + rng.normal(0.0, 1.1, len(timestamps))
    cloud = np.clip(4.0 + 2.8 * np.sin(2.0 * np.pi * day_index / 6.0) + rng.normal(0.0, 1.3, len(timestamps)), 0, 8)
    weather = pd.DataFrame(
        {
            "site_id": 0,
            "timestamp": timestamps,
            "air_temperature": outdoor_temperature,
            "dew_temperature": outdoor_temperature - rng.uniform(2.0, 8.0, len(timestamps)),
            "cloud_coverage": cloud,
            "sea_level_pressure": 1013.0 + rng.normal(0.0, 5.0, len(timestamps)),
            "wind_direction": rng.uniform(0.0, 360.0, len(timestamps)),
            "wind_speed": np.clip(rng.gamma(shape=2.0, scale=1.8, size=len(timestamps)), 0, 18),
        }
    )

    building_ids = np.arange(cfg.n_buildings, dtype=int)
    square_feet = rng.integers(18000, 85000, size=cfg.n_buildings)
    metadata = pd.DataFrame(
        {
            "site_id": 0,
            "building_id": building_ids,
            "primary_use": ["Education", "Office", "Lodging", "Public services"][: cfg.n_buildings],
            "square_feet": square_feet,
            "year_built": rng.integers(1970, 2015, size=cfg.n_buildings),
            "floor_count": rng.integers(2, 9, size=cfg.n_buildings),
        }
    )

    business_hours = ((hour >= 7) & (hour <= 19) & (day < 5)).astype(float)
    shoulder_hours = ((hour >= 6) & (hour <= 21)).astype(float)
    rows: list[pd.DataFrame] = []
    for index, building_id in enumerate(building_ids):
        size_scale = square_feet[index] / 10000.0
        base_load = 7.0 + 1.6 * size_scale + 2.0 * index
        occupancy = business_hours * (0.8 + 0.12 * index) + 0.13 * shoulder_hours
        heating = np.maximum(17.0 - outdoor_temperature, 0.0) * (0.23 + 0.015 * index) * size_scale
        cooling = np.maximum(outdoor_temperature - 22.0, 0.0) * (0.31 + 0.018 * index) * size_scale
        daylight = np.maximum(1.0 - cloud / 8.0, 0.0) * np.maximum(np.sin(np.pi * (hour - 6) / 12.0), 0.0)
        energy = base_load + occupancy * (3.4 + 0.35 * size_scale) + heating + cooling - 0.35 * daylight
        energy += rng.normal(0.0, 0.45 + 0.06 * size_scale, len(timestamps))
        spike_mask = rng.random(len(timestamps)) < 0.006
        energy[spike_mask] *= rng.uniform(1.25, 1.75, spike_mask.sum())
        rows.append(
            pd.DataFrame(
                {
                    "building_id": building_id,
                    "meter": 0,
                    "timestamp": timestamps,
                    "meter_reading": np.clip(energy, 0.05, None),
                    "synthetic_occupancy_proxy": occupancy,
                    "synthetic_energy_spike": spike_mask.astype(int),
                }
            )
        )
    train = pd.concat(rows, ignore_index=True)
    return train, weather, metadata


def write_synthetic_demo_data(destination: str | Path, config: SyntheticCampusConfig | None = None) -> dict[str, Path]:
    """Write an ASHRAE-like synthetic dataset to a local ignored directory."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    train, weather, metadata = generate_synthetic_ashrae_like_data(config)
    paths = {
        "train": root / "train.csv",
        "weather": root / "weather_train.csv",
        "metadata": root / "building_metadata.csv",
    }
    train.to_csv(paths["train"], index=False)
    weather.to_csv(paths["weather"], index=False)
    metadata.to_csv(paths["metadata"], index=False)
    return paths
