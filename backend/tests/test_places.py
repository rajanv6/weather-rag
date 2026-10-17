import numpy as np

import places
import rag
from data import Snapshot, WeatherDoc


def test_detect_cities_resolves_aliases_in_order_of_mention():
    found = places.detect_cities("Compare Bengaluru with New York City, then Madras")
    assert found == ["Bangalore", "New York", "Chennai"]


def test_detect_cities_ignores_accents_and_punctuation():
    assert places.detect_cities("rain in Sao Paulo?!") == ["Sao Paulo"]
    assert places.detect_cities("rain in São Paulo?!") == ["Sao Paulo"]


def test_detect_cities_needs_whole_words():
    assert places.detect_cities("my chrome browser in Perthshire") == []  # not Rome, not Perth


def test_resolve_city_uses_the_built_in_list_without_a_network_call(monkeypatch):
    monkeypatch.setattr(places, "geocode", lambda name: (_ for _ in ()).throw(AssertionError("no network")))
    assert places.resolve_city("Madras")[0] == "Chennai"
    assert places.resolve_city("tokyo")[:3] == ("Tokyo", "Japan", "Asia")


def test_resolve_city_falls_back_to_the_geocoder(monkeypatch):
    monkeypatch.setattr(places, "geocode", lambda name: ("Pune", "India", "Maharashtra", 18.5, 73.8))
    assert places.resolve_city("Pune") == ("Pune", "India", "Maharashtra", 18.5, 73.8)
    monkeypatch.setattr(places, "geocode", lambda name: None)
    assert places.resolve_city("Atlantis") is None


# --- retrieve(): city questions get that city's notes; everything else gets none -----------

def make_index(monkeypatch):
    docs = [
        WeatherDoc(city, "Asia", kind, f"{city} {kind}")
        for city in ("Chennai", "Delhi")
        for kind in ("year_ago", "forecast", "past_week")  # deliberately shuffled
    ]
    monkeypatch.setattr(rag, "_snapshot", Snapshot(docs=docs, raw={}))
    monkeypatch.setattr(rag, "_vectors", np.eye(len(docs)))
    embedded = []
    monkeypatch.setattr(rag, "embed", lambda texts: embedded.append(texts) or np.ones((1, len(docs))))
    return embedded


def test_retrieve_returns_all_notes_for_each_named_city_in_mention_order(monkeypatch):
    make_index(monkeypatch)
    got = rag.retrieve("Is it wetter in Delhi or Chennai?")
    assert [(d.city, d.kind) for d in got] == [
        ("Delhi", "forecast"), ("Delhi", "past_week"), ("Delhi", "year_ago"),
        ("Chennai", "forecast"), ("Chennai", "past_week"), ("Chennai", "year_ago"),
    ]


def test_retrieve_returns_nothing_without_a_city(monkeypatch):
    embedded = make_index(monkeypatch)
    assert rag.retrieve("which Asian cities are rainiest?") == []
    assert embedded == []  # didn't even embed the question
