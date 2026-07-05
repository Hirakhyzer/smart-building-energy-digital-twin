"""Run a chronological energy-demand forecast after local data inspection."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buildingtwin.config import ensure_output_dirs, load_config, set_seed
from buildingtwin.data_io import add_time_features, build_energy_frame, load_ashrae_tables
from buildingtwin.energy_model import build_features, evaluate, fit_baseline
from buildingtwin.plots import forecast_plot
from buildingtwin.splits import assert_temporal_order, chronological_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run building energy forecasting baseline.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    selected = config["data"].get("building_ids")
    if not selected:
        raise ValueError("Set data.building_ids in configs/base.yaml after inspecting data. This avoids an arbitrary building subset.")
    train, weather, metadata = load_ashrae_tables(config["paths"]["raw_dir"])
    frame = build_energy_frame(train, weather, metadata, meter_type=config["data"].get("meter_type"))
    frame = frame.loc[frame["building_id"].isin(selected)].copy()
    frame = frame.loc[frame[config["forecast"]["target_column"]].notna()].copy()
    frame = add_time_features(frame)
    if len(frame) > int(config["forecast"]["max_rows"]):
        raise ValueError("Selected data exceed forecast.max_rows. Choose fewer buildings or raise the limit after checking memory/runtime.")
    train_frame, validation_frame, test_frame = chronological_split(frame, config["data"]["validation_fraction"], config["data"]["test_fraction"])
    assert_temporal_order(train_frame, validation_frame, test_frame)
    x_train, feature_names = build_features(train_frame)
    x_validation, _ = build_features(validation_frame)
    x_test, _ = build_features(test_frame)
    target = config["forecast"]["target_column"]
    model = fit_baseline(pd.concat([x_train, x_validation]), pd.concat([train_frame[target], validation_frame[target]]), int(config["seed"]))
    predicted = model.predict(x_test)
    metrics = evaluate(test_frame[target].to_numpy(), predicted)
    outputs = ensure_output_dirs(config["paths"]["output_dir"])
    result = {
        "experiment": "hist_gradient_boosting_energy_forecast",
        "seed": config["seed"],
        "building_ids": selected,
        "meter_type": config["data"].get("meter_type"),
        "feature_names": feature_names,
        "rows": {"train": len(train_frame), "validation": len(validation_frame), "test": len(test_frame)},
        "metrics": metrics.__dict__,
    }
    (outputs["results"] / "forecast_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    preview = test_frame[["timestamp", target]].copy()
    preview["prediction"] = predicted
    preview.to_csv(outputs["results"] / "forecast_predictions.csv", index=False)
    forecast_plot(preview, "timestamp", target, "prediction", outputs["figures"] / "forecast_window.png")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
