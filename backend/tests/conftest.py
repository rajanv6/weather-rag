"""Shared fixtures. Nothing here touches the network, the LLM or the embedding model."""

import pytest

import rag
import tools
from data import Snapshot


def series(mm, hi=25.0, lo=15.0):
    """A minimal Open-Meteo `daily` block for the given precipitation list."""
    n = len(mm)
    return {
        "time": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "weather_code": [0] * n,
        "temperature_2m_max": [hi] * n,
        "temperature_2m_min": [lo] * n,
        "precipitation_sum": list(mm),
    }


def forecast(past_mm, future_mm, hi=25.0):
    """14 days like the real forecast call: 7 past days then today + 6 ahead."""
    assert len(past_mm) == len(future_mm) == 7
    return {"daily": series(list(past_mm) + list(future_mm), hi=hi)}


@pytest.fixture
def snapshot(monkeypatch):
    """Four cities with hand-picked rain so rankings have known answers.

    next week:  Chennai 7 rain days/35 mm, Colombo 7/14 mm, Delhi 0/6.3 mm (0.9 mm
    days are not rain days), London 2/20 mm.   Delhi has no year-ago data."""
    raw = {  # Colombo listed before Chennai so a stable sort alone can't fake the tie-break
        "Colombo": {"forecast": forecast([1] * 7, [2] * 7, hi=30), "archive": {"daily": series([0] * 7)}},
        "Chennai": {"forecast": forecast([0] * 7, [5] * 7, hi=31), "archive": {"daily": series([3, 0, 0, 0, 0, 0, 0])}},
        "Delhi": {"forecast": forecast([0] * 7, [0.9] * 7, hi=35), "archive": None},
        "London": {
            "forecast": forecast([0] * 7, [10, 10, 0, 0, 0, 0, 0], hi=14),
            "archive": {"daily": series([1] * 7)},
        },
    }
    monkeypatch.setattr(rag, "_snapshot", Snapshot(docs=[], raw=raw))
    return raw


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    tools._range_cache.clear()
    yield
    tools._range_cache.clear()
