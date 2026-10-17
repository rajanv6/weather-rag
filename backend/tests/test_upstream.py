"""The Open-Meteo politeness layer in data.py: backoff, spacing, daily budget, geocode cache."""

from datetime import date, timedelta

import pytest

import data


class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body if body is not None else [{"ok": True}], headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def no_waiting(monkeypatch):
    """Skip real sleeping and spacing; record the sleeps _get asks for."""
    sleeps = []
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    monkeypatch.setattr(data, "_wait_turn", lambda: None)
    return sleeps


def script_responses(monkeypatch, responses):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(data.httpx, "get", fake_get)
    return calls


def test_get_honours_retry_after_on_429(monkeypatch, no_waiting):
    calls = script_responses(monkeypatch, [Resp(429, headers={"Retry-After": "7"}), Resp(200, [{"a": 1}])])
    assert data._get("http://x", {}) == [{"a": 1}]
    assert len(calls) == 2 and no_waiting == [7.0]


def test_get_backs_off_exponentially_without_retry_after(monkeypatch, no_waiting):
    script_responses(monkeypatch, [Resp(429), Resp(429), Resp(200)])
    data._get("http://x", {})
    assert no_waiting == [2, 4]


def test_get_gives_up_after_four_attempts(monkeypatch, no_waiting):
    calls = script_responses(monkeypatch, [Resp(429)])
    with pytest.raises(RuntimeError, match="429"):
        data._get("http://x", {})
    assert len(calls) == 4


def test_get_wraps_a_single_object_in_a_list(monkeypatch, no_waiting):
    script_responses(monkeypatch, [Resp(200, {"daily": {}})])
    assert data._get("http://x", {}) == [{"daily": {}}]


def test_calls_are_spaced_out(monkeypatch):
    sleeps = []
    monkeypatch.setattr(data.time, "sleep", sleeps.append)
    monkeypatch.setattr(data.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(data, "_last_call", 100.0)  # a call just happened
    data._wait_turn()
    assert sleeps == [pytest.approx(data.MIN_INTERVAL_S)]


@pytest.fixture
def fresh_budget(monkeypatch):
    monkeypatch.setattr(data, "ON_DEMAND_PER_DAY", 2)
    monkeypatch.setattr(data, "_budget_used", 0)
    monkeypatch.setattr(data, "_budget_day", date.today())


def test_on_demand_calls_stop_at_the_daily_budget(monkeypatch, no_waiting, fresh_budget):
    script_responses(monkeypatch, [Resp(200)])
    data._get("http://x", {}, on_demand=True)
    data._get("http://x", {}, on_demand=True)
    with pytest.raises(data.UpstreamBudgetError, match="daily limit"):
        data._get("http://x", {}, on_demand=True)


def test_scheduled_refresh_calls_do_not_spend_the_budget(monkeypatch, no_waiting, fresh_budget):
    script_responses(monkeypatch, [Resp(200)])
    for _ in range(10):
        data._get("http://x", {})  # on_demand defaults to False
    data._get("http://x", {}, on_demand=True)  # budget untouched


def test_budget_resets_on_a_new_day(monkeypatch, no_waiting, fresh_budget):
    script_responses(monkeypatch, [Resp(200)])
    data._get("http://x", {}, on_demand=True)
    data._get("http://x", {}, on_demand=True)
    monkeypatch.setattr(data, "_budget_day", date.today() - timedelta(days=1))
    data._get("http://x", {}, on_demand=True)  # new day, allowed again


def test_geocode_results_are_cached(monkeypatch):
    calls = []

    def fake_get(url, params, on_demand=False):
        calls.append(params["name"])
        hit = {"name": "Pune", "country": "India", "admin1": "Maharashtra", "latitude": 18.5, "longitude": 73.8}
        return [{"results": [hit]}]

    monkeypatch.setattr(data, "_get", fake_get)
    data.geocode.cache_clear()
    assert data.geocode("Pune") == ("Pune", "India", "Maharashtra", 18.5, 73.8)
    data.geocode("Pune")
    assert calls == ["Pune"]
    data.geocode.cache_clear()
