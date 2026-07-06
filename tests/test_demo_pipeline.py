from buildingtwin.data_io import add_time_features, build_energy_frame
from buildingtwin.energy_model import build_features, fit_baseline
from buildingtwin.splits import assert_temporal_order, chronological_split
from buildingtwin.synthetic import SyntheticCampusConfig, generate_synthetic_ashrae_like_data


def test_synthetic_data_runs_through_forecast_pipeline():
    train, weather, metadata = generate_synthetic_ashrae_like_data(SyntheticCampusConfig(days=14, n_buildings=2, seed=7))
    frame = add_time_features(build_energy_frame(train, weather, metadata, meter_type=0))
    train_frame, validation_frame, test_frame = chronological_split(frame, validation_fraction=0.2, test_fraction=0.2)
    assert_temporal_order(train_frame, validation_frame, test_frame)
    x_train, features = build_features(train_frame)
    x_test, _ = build_features(test_frame)
    model = fit_baseline(x_train, train_frame["meter_reading"], seed=7)
    prediction = model.predict(x_test)
    assert features
    assert len(prediction) == len(test_frame)
