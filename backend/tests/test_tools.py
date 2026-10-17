import json
from datetime import date, timedelta

import climate
import tools

from conftest import series

PUNE = ("Pune", "India", "Maharashtra", 18.52, 73.86)


def cities(result):
    return [r["city"] for r in result["results"]]


# --- rank_cities ---------------------------------------------------------------

def test_rank_cities_filters_by_region_and_breaks_ties_on_rainfall(snapshot):
    out = tools.rank_cities("rain_days", "next_week", region="Asia")
    # Chennai and Colombo both have 7 rain days; Chennai wins on total mm. London is Europe.
    assert cities(out) == ["Chennai", "Colombo", "Delhi"]
    assert out["results"][0]["rain_days"] == 7
    assert out["cities_considered"] == 3


def test_rank_cities_lowest_order(snapshot):
    out = tools.rank_cities("rain_days", "next_week", region="Asia", order="lowest")
    assert cities(out)[0] == "Delhi"  # 0.9 mm/day is below the 1 mm rain-day threshold


def test_rank_cities_region_is_forgiving(snapshot):
    assert cities(tools.rank_cities("rain_days", "next_week", region="asian")) == cities(
        tools.rank_cities("rain_days", "next_week", region="Asia")
    )


def test_rank_cities_windows_use_different_days(snapshot):
    past = tools.rank_cities("rain_days", "past_week")
    assert past["results"][0]["city"] == "Colombo"  # 1 mm every past day
    assert tools.rank_cities("rain_days", "next_week")["results"][0]["city"] == "Chennai"


def test_rank_cities_skips_cities_without_year_ago_data(snapshot):
    out = tools.rank_cities("rain_mm", "year_ago_week")
    assert "Delhi" not in cities(out)
    assert out["cities_considered"] == 3


def test_rank_cities_temperature_metric_and_limit(snapshot):
    out = tools.rank_cities("avg_high_c", "next_week", limit=1)
    assert cities(out) == ["Delhi"]
    assert out["results"][0]["avg_high_c"] == 35.0


# --- get_weather_range -----------------------------------------------------------

def fake_range(mm_for):
    def fetch(meta, start, end):
        n = (end - start).days + 1
        return series(mm_for(n))
    return fetch


def test_get_weather_range_counts_rain_days_at_the_threshold(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: PUNE)
    monkeypatch.setattr(tools, "_fetch_range", lambda m, s, e: series([0.9, 1.0, 1.1, 0.0]))
    out = tools.get_weather_range("Pune", "2025-12-24", "2025-12-27")
    assert out["rain_days"] == 2  # 1.0 and 1.1 count, 0.9 does not
    assert out["rain_mm"] == 3.0
    assert out["city"] == "Pune, India"
    assert len(out["daily"]) == 4


def test_get_weather_range_omits_daily_list_for_long_ranges(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: PUNE)
    monkeypatch.setattr(tools, "_fetch_range", fake_range(lambda n: [2.0] * n))
    out = tools.get_weather_range("Pune", "2025-01-01", "2025-01-31")
    assert out["days"] == 31 and out["rain_days"] == 31
    assert "daily" not in out


def test_get_weather_range_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: PUNE)
    monkeypatch.setattr(tools, "_fetch_range", lambda *a: pytest_fail())
    today = date.today()
    assert "before" in tools.get_weather_range("Pune", "2025-02-02", "2025-02-01")["error"]
    assert "too long" in tools.get_weather_range("Pune", "2020-01-01", "2021-12-31")["error"]
    far = (today + timedelta(days=tools.MAX_FORECAST_AHEAD + 1)).isoformat()
    assert "ahead" in tools.get_weather_range("Pune", today.isoformat(), far)["error"]


def test_get_weather_range_unknown_city(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: None)
    assert "could not find" in tools.get_weather_range("Atlantis", "2025-01-01", "2025-01-02")["error"]


def pytest_fail():
    raise AssertionError("validation should have stopped the request before any fetch")


# --- _fetch_range: which API serves which dates, and caching ---------------------------

def recording_get(calls):
    def get(url, params, on_demand=False):
        calls.append((url, params, on_demand))
        start, end = date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"])
        days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
        return [{"daily": {**series([1.0] * len(days)), "time": days}}]
    return get


def test_fetch_range_old_dates_use_only_the_archive(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get", recording_get(calls))
    start = date.today() - timedelta(days=400)
    tools._fetch_range(PUNE, start, start + timedelta(days=9))
    assert len(calls) == 1 and "archive" in calls[0][0]
    assert calls[0][2] is True  # counted against the daily on-demand budget


def test_fetch_range_recent_and_future_dates_use_only_the_forecast(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get", recording_get(calls))
    tools._fetch_range(PUNE, date.today(), date.today() + timedelta(days=2))
    assert len(calls) == 1 and "archive" not in calls[0][0]


def test_fetch_range_spanning_the_cutoff_merges_both_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get", recording_get(calls))
    start, end = date.today() - timedelta(days=20), date.today() + timedelta(days=2)
    daily = tools._fetch_range(PUNE, start, end)
    assert len(calls) == 2
    assert daily["time"] == [(start + timedelta(days=i)).isoformat() for i in range(23)]  # no gap, no overlap


def test_fetch_range_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get", recording_get(calls))
    start = date.today() - timedelta(days=400)
    for _ in range(3):
        tools._fetch_range(PUNE, start, start + timedelta(days=4))
    assert len(calls) == 1


# --- travel_advice ----------------------------------------------------------------

def climate_table():
    """Same 25C highs all year; rain days grow 0,2,4..22 so Jan is best and Dec worst."""
    return [
        {"month": m, "avg_high_c": 25.0, "avg_low_c": 18.0, "rain_days": i * 2.0, "rain_mm": i * 20}
        for i, m in enumerate(climate.MONTHS)
    ]


def test_travel_advice_ranks_months_by_comfort(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: PUNE)
    monkeypatch.setattr(climate, "normals", lambda lat, lon: climate_table())
    monkeypatch.setattr(tools, "_fetch_range", lambda *a: series([0.0] * 7))
    out = tools.travel_advice("Pune", month="March")
    assert [m["month"] for m in out["best_months"]] == ["Jan", "Feb", "Mar"]
    assert out["worst_month"]["month"] == "Dec"
    assert out["month_asked"]["month"] == "Mar" and out["month_asked"]["rank_of_12"] == 3
    assert "20-28C" in out["scoring"]  # the model is told the formula so it can explain it


def test_travel_advice_compares_forecast_only_for_the_current_month(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: PUNE)
    monkeypatch.setattr(climate, "normals", lambda lat, lon: climate_table())
    monkeypatch.setattr(tools, "_fetch_range", lambda *a: series([5.0, 5.0, 0, 0, 0, 0, 0]))

    now = tools.travel_advice("Pune")
    assert now["next_7_days_forecast"]["rain_days"] == 2
    assert "normal_rain_days_per_7" in now["next_7_days_forecast"]

    other = climate.MONTHS[date.today().month % 12]
    assert "next_7_days_forecast" not in tools.travel_advice("Pune", month=other)


def test_travel_advice_unknown_city(monkeypatch):
    monkeypatch.setattr(tools, "resolve_city", lambda name: None)
    assert "error" in tools.travel_advice("Atlantis")


# --- run(): the model-facing wrapper ---------------------------------------------------

def test_run_returns_json_for_a_good_call(snapshot):
    out = json.loads(tools.run("rank_cities", json.dumps({"metric": "rain_mm", "window": "next_week"})))
    assert out["results"][0]["city"] == "Chennai"


def test_run_turns_failures_into_error_json_instead_of_raising():
    assert "error" in json.loads(tools.run("no_such_tool", "{}"))
    assert "error" in json.loads(tools.run("rank_cities", "not json"))
    assert "error" in json.loads(tools.run("rank_cities", json.dumps({"bogus_argument": 1})))
