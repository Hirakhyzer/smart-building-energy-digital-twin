"""Run a clearly labeled model-injected degradation scenario."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buildingtwin.config import ensure_output_dirs, load_config
from buildingtwin.data_io import add_time_features, build_energy_frame, load_ashrae_tables
from buildingtwin.degradation import DegradationScenario, inject_degradation_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simulated HVAC degradation study inputs.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    selected = config["data"].get("building_ids")
    if not selected:
        raise ValueError("Set data.building_ids after inspecting the local data.")
    train, weather, metadata = load_ashrae_tables(config["paths"]["raw_dir"])
    frame = add_time_features(build_energy_frame(train, weather, metadata, config["data"].get("meter_type")))
    subset = frame.loc[frame["building_id"] == selected[0]].sort_values("timestamp").copy()
    scenario_cfg = config["degradation_scenario"]
    scenario = DegradationScenario(scenario_cfg["efficiency_drop_fraction"], scenario_cfg["onset_fraction"], scenario_cfg["sensor_bias"])
    output = inject_degradation_scenario(subset, energy_column=config["forecast"]["target_column"], scenario=scenario)
    outputs = ensure_output_dirs(config["paths"]["output_dir"])
    output.to_csv(outputs["results"] / "simulated_degradation_scenario.csv", index=False)
    print("Wrote simulated scenario only; this is not a measured HVAC fault dataset.")

if __name__ == "__main__":
    main()
