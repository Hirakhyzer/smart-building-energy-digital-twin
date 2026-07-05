"""Figures generated from local executed experiments."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def forecast_plot(table: pd.DataFrame, time_col: str, actual_col: str, prediction_col: str, path: str | Path) -> None:
    """Save a time-series comparison of observed and predicted energy."""
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(table[time_col], table[actual_col], label="Observed")
    axis.plot(table[time_col], table[prediction_col], label="Predicted")
    axis.set(xlabel="Time", ylabel="Energy reading", title="Energy forecast evaluation")
    axis.legend(); axis.grid(True, alpha=0.25)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=250); plt.close(figure)

def scenario_plot(table: pd.DataFrame, path: str | Path) -> None:
    """Save the comfort-cost trade-off from evaluated targets."""
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    axis.scatter(table["cost_proxy"], table["comfort_degree_hours"])
    axis.set(xlabel="Energy cost proxy", ylabel="Comfort degree-hours", title="Thermal scenario trade-off")
    axis.grid(True, alpha=0.25)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=250); plt.close(figure)
