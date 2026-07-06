"""Run a complete local smart-building digital-twin demonstration.

This command generates a deterministic synthetic dataset, then exercises the same
forecasting, thermal-scenario, degradation, and anomaly components that will be
used later with downloaded ASHRAE files. Every output is labelled synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from buildingtwin.anomalies import build_residual_features, fit_anomaly_detector, score_anomalies
from buildingtwin.config import ensure_output_dirs, set_seed
from buildingtwin.data_io import add_time_features, build_energy_frame, load_ashrae_tables
from buildingtwin.degradation import DegradationScenario, inject_degradation_scenario
from buildingtwin.demo_plots import plot_anomaly_scores, plot_energy_weather_profile, plot_simulated_efficiency
from buildingtwin.energy_model import build_features, evaluate, fit_baseline
from buildingtwin.plots import forecast_plot, scenario_plot
from buildingtwin.scenarios import choose_best_target
from buildingtwin.splits import assert_temporal_order, chronological_split
from buildingtwin.synthetic import SyntheticCampusConfig, write_synthetic_demo_data
from buildingtwin.thermal import RCParameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic synthetic smart-building digital-twin demo.")
    parser.add_argument("--days", type=int, default=90, help="Synthetic hourly history length; minimum 14 days.")
    parser.add_argument("--buildings", type=int, default=4, help="Number of synthetic buildings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--demo-data-dir", default="data/interim/synthetic_ashrae")
    args = parser.parse_args()

    set_seed(args.seed)
    synthetic_config = SyntheticCampusConfig(days=args.days, n_buildings=args.buildings, seed=args.seed)
    write_synthetic_demo_data(args.demo_data_dir, synthetic_config)
    train, weather, metadata = load_ashrae_tables(args.demo_data_dir)
    joined = add_time_features(build_energy_frame(train, weather, metadata, meter_type=0))
    train_frame, validation_frame, test_frame = chronological_split(joined, validation_fraction=0.15, test_fraction=0.15)
    assert_temporal_order(train_frame, validation_frame, test_frame)

    x_train, feature_names = build_features(train_frame)
    x_validation, _ = build_features(validation_frame)
    x_test, _ = build_features(test_frame)
    model = fit_baseline(x_train, train_frame["meter_reading"], seed=args.seed)
    validation_prediction = model.predict(x_validation)
    validation_metrics = evaluate(validation_frame["meter_reading"].to_numpy(), validation_prediction)
    final_model = fit_baseline(pd.concat([x_train, x_validation]), pd.concat([train_frame["meter_reading"], validation_frame["meter_reading"]]), seed=args.seed)
    test_prediction = final_model.predict(x_test)
    test_metrics = evaluate(test_frame["meter_reading"].to_numpy(), test_prediction)

    selected_building = int(joined["building_id"].min())
    validation_view = validation_frame.loc[validation_frame["building_id"] == selected_building, ["timestamp", "meter_reading"]].copy()
    validation_view["prediction"] = validation_prediction[validation_frame["building_id"].to_numpy() == selected_building]
    test_view = test_frame.loc[test_frame["building_id"] == selected_building, ["timestamp", "meter_reading", "air_temperature", "hour"]].copy()
    test_view["prediction"] = test_prediction[test_frame["building_id"].to_numpy() == selected_building]

    validation_residuals = build_residual_features(validation_view, "meter_reading", "prediction")
    test_residuals = build_residual_features(test_view, "meter_reading", "prediction")
    anomaly_model = fit_anomaly_detector(validation_residuals, seed=args.seed)
    anomaly_scores = score_anomalies(anomaly_model, test_residuals)
    anomaly_scores.insert(0, "timestamp", test_view["timestamp"].to_numpy())

    horizon = min(168, len(test_view))
    scenario_weather = test_view["air_temperature"].to_numpy()[:horizon]
    synthetic_price = np.where(test_view["hour"].to_numpy()[:horizon] >= 16, 1.5, 0.8)
    parameters = RCParameters(r_ao=2.5, r_am=1.0, c_air=2.0, c_mass=12.0, hvac_efficiency=1.0)
    best_scenario, scenario_scores = choose_best_target(
        scenario_weather,
        synthetic_price,
        candidates=[20.0, 21.0, 22.0, 23.0, 24.0],
        comfort_min=20.0,
        comfort_max=24.0,
        parameters=parameters,
    )
    scenario_table = pd.DataFrame([score.__dict__ for score in scenario_scores])

    degradation = inject_degradation_scenario(
        test_view,
        energy_column="meter_reading",
        temperature_column=None,
        scenario=DegradationScenario(efficiency_drop_fraction=0.18, onset_fraction=0.45, sensor_bias=0.0),
    )

    outputs = ensure_output_dirs(args.output_dir)
    profile_window = joined.loc[joined["building_id"] == selected_building].head(min(336, len(joined))).copy()
    plot_energy_weather_profile(profile_window, outputs["figures"] / "synthetic_energy_weather_profile.png")
    forecast_plot(test_view, "timestamp", "meter_reading", "prediction", outputs["figures"] / "synthetic_forecast_window.png")
    plot_anomaly_scores(anomaly_scores, outputs["figures"] / "synthetic_anomaly_scores.png")
    scenario_plot(scenario_table, outputs["figures"] / "synthetic_comfort_cost_tradeoff.png")
    plot_simulated_efficiency(degradation, outputs["figures"] / "synthetic_degradation_profile.png")

    test_view.to_csv(outputs["results"] / "synthetic_demo_predictions.csv", index=False)
    anomaly_scores.to_csv(outputs["results"] / "synthetic_demo_anomaly_scores.csv", index=False)
    scenario_table.to_csv(outputs["results"] / "synthetic_demo_thermal_scenarios.csv", index=False)
    degradation.to_csv(outputs["results"] / "synthetic_demo_degradation.csv", index=False)
    summary = {
        "data_origin": "synthetic; generated locally by buildingtwin.synthetic",
        "seed": args.seed,
        "synthetic_config": synthetic_config.__dict__,
        "selected_building": selected_building,
        "feature_names": feature_names,
        "rows": {"train": len(train_frame), "validation": len(validation_frame), "test": len(test_frame)},
        "validation_metrics": validation_metrics.__dict__,
        "test_metrics": test_metrics.__dict__,
        "thermal_price_note": "Synthetic peak/off-peak proxy; not a utility tariff.",
        "best_thermal_scenario": best_scenario.__dict__,
        "anomaly_note": "Unsupervised residual anomaly evidence; synthetic demo only.",
        "degradation_note": "Model-injected efficiency decline; not observed HVAC failure ground truth.",
    }
    (outputs["results"] / "synthetic_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
