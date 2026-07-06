from buildingtwin.synthetic import SyntheticCampusConfig, generate_synthetic_ashrae_like_data


def test_synthetic_generator_has_consistent_hourly_tables():
    train, weather, metadata = generate_synthetic_ashrae_like_data(SyntheticCampusConfig(days=14, n_buildings=3, seed=5))
    assert len(weather) == 14 * 24
    assert len(metadata) == 3
    assert len(train) == 14 * 24 * 3
    assert set(["building_id", "meter", "timestamp", "meter_reading"]).issubset(train.columns)
    assert (train["meter_reading"] > 0).all()


def test_synthetic_generator_is_reproducible_for_fixed_seed():
    config = SyntheticCampusConfig(days=14, n_buildings=2, seed=13)
    first, _, _ = generate_synthetic_ashrae_like_data(config)
    second, _, _ = generate_synthetic_ashrae_like_data(config)
    assert first.equals(second)
