"""Additional figures for the synthetic end-to-end demonstration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _destination(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_energy_weather_profile(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot synthetic meter demand and outdoor temperature for one building."""
    figure, left = plt.subplots(figsize=(10.5, 4.6))
    left.plot(frame["timestamp"], frame["meter_reading"], label="Synthetic meter reading")
    left.set(xlabel="Time", ylabel="Synthetic energy reading", title="Synthetic building energy and weather profile")
    right = left.twinx()
    right.plot(frame["timestamp"], frame["air_temperature"], label="Outdoor temperature")
    right.set_ylabel("Outdoor temperature")
    handles_a, labels_a = left.get_legend_handles_labels()
    handles_b, labels_b = right.get_legend_handles_labels()
    left.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    left.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(_destination(path), dpi=250)
    plt.close(figure)


def plot_anomaly_scores(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot residual anomaly evidence from a local executed demonstration."""
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    axis.plot(frame["timestamp"], frame["anomaly_score"], label="Residual anomaly score")
    axis.set(xlabel="Time", ylabel="Anomaly score", title="Forecast-residual anomaly evidence")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_destination(path), dpi=250)
    plt.close(figure)


def plot_simulated_efficiency(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot the injected synthetic efficiency profile with clear simulated labeling."""
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    axis.plot(frame["timestamp"], frame["simulated_efficiency"], label="Simulated efficiency multiplier")
    axis.set(xlabel="Time", ylabel="Efficiency multiplier", title="Simulated HVAC efficiency-decline scenario", ylim=(0, 1.05))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_destination(path), dpi=250)
    plt.close(figure)
