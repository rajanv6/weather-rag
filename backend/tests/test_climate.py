import climate


def test_comfort_score_in_the_ideal_band_is_perfect():
    assert climate.comfort_score({"avg_high_c": 24, "rain_days": 0}) == 100


def test_comfort_score_penalises_heat_cold_and_rain():
    assert climate.comfort_score({"avg_high_c": 32, "rain_days": 0}) == 84  # 4 degC over x 4
    assert climate.comfort_score({"avg_high_c": 10, "rain_days": 0}) == 60  # 10 degC under x 4
    assert climate.comfort_score({"avg_high_c": 24, "rain_days": 10}) == 75  # 2.5 per rain day


def test_comfort_score_never_goes_below_zero():
    assert climate.comfort_score({"avg_high_c": 45, "rain_days": 30}) == 0


def fake_archive(years):
    """One day per month per year: 30C/20C and 2 mm of rain (a rain day)."""
    days = [f"{y}-{m:02d}-15" for y in years for m in range(1, 13)]
    n = len(days)
    return [{"daily": {
        "time": days,
        "temperature_2m_max": [30.0] * n,
        "temperature_2m_min": [20.0] * n,
        "precipitation_sum": [2.0] * n,
    }}]


def test_normals_average_over_the_years_and_are_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(climate, "CACHE_FILE", tmp_path / "climate.json")
    calls = []

    def get(url, params, on_demand=False):
        calls.append(params)
        return fake_archive(range(2020, 2025))

    monkeypatch.setattr(climate, "_get", get)
    table = climate.normals(13.08, 80.27)
    assert len(table) == 12 and table[0]["month"] == "Jan"
    # 1 rain day per month per year, averaged over YEARS=5 (the fake supplies 5 years) -> 1.0
    assert table[0] == {"month": "Jan", "avg_high_c": 30.0, "avg_low_c": 20.0, "rain_days": 1.0, "rain_mm": 2}

    assert climate.normals(13.08, 80.27) == table
    assert len(calls) == 1  # second lookup came from the disk cache
