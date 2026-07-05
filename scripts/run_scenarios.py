"""Run RC thermal scenarios from selected building weather observations."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from buildingtwin.config import ensure_output_dirs, load_config, set_seed
from buildingtwin.data_io import add_time_features, build_energy_frame, load_ashrae_tables
from buildingtwin.plots import scenario_plot
from buildingtwin.scenarios import choose_best_target
from buildingtwin.thermal import RCParameters

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RC thermal scenarios.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--horizon", type=int, default=168)
    args = parser.parse_args()
    config = load_config(args.config); set_seed(int(config["seed"]))
    selected = config["data"].get("building_ids")
    if not selected:
        raise ValueError("Set data.building_ids after inspection before running scenarios.")
    train, weather, metadata = load_ashrae_tables(config["paths"]["raw_dir"])
    frame = add_time_features(build_energy_frame(train, weather, metadata, config["data"].get("meter_type")))
    chosen = frame.loc[frame["building_id"] == selected[0]].sort_values("timestamp").head(args.horizon).copy()
    if "air_temperature" not in chosen.columns or len(chosen) < 24:
        raise ValueError("Need air_temperature and at least 24 observations for an RC scenario.")
    price = np.where(chosen["hour"].between(16, 20), 1.4, 1.0)
    thermal = config["thermal_model"]
    parameters = RCParameters(thermal["r_ao"], thermal["r_am"], thermal["c_air"], thermal["c_mass"], thermal["hvac_efficiency"])
    best, scores = choose_best_target(chosen["air_temperature"].to_numpy(), price, thermal["candidate_setpoints"], thermal["comfort_min"], thermal["comfort_max"], parameters)
    results = pd.DataFrame([score.__dict__ for score in scores])
    outputs = ensure_output_dirs(config["paths"]["output_dir"])
    results.to_csv(outputs["results"] / "thermal_scenarios.csv", index=False)
    scenario_plot(results, outputs["figures"] / "comfort_cost_tradeoff.png")
    summary = {"building_id": selected[0], "horizon": len(chosen), "price_note": "Synthetic peak/off-peak proxy", "best": best.__dict__}
    (outputs["results"] / "thermal_scenario_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
