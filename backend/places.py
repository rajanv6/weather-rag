"""Turning free text into cities and regions (keyword matching, no model)."""

import re
import unicodedata

from data import CITIES, geocode

CITY_ALIASES = {
    "bengaluru": "Bangalore", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
    "nyc": "New York", "new york city": "New York", "washington": "Washington DC",
    "saigon": "Ho Chi Minh City", "hcmc": "Ho Chi Minh City", "rio": "Rio de Janeiro",
    "kiev": "Kyiv", "sf": "San Francisco",
}
REGION_KEYWORDS = {
    "north america": "North America", "south america": "South America", "latin america": "South America",
    "europe": "Europe", "european": "Europe", "asia": "Asia", "asian": "Asia",
    "africa": "Africa", "african": "Africa", "australia": "Oceania", "oceania": "Oceania",
}
_BY_NAME = {c[0]: c for c in CITIES}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


_NAME_KEYS = {norm(c[0]): c[0] for c in CITIES} | CITY_ALIASES


def detect_cities(query: str) -> list[str]:
    q = f" {norm(query)} "
    first_seen: dict[str, int] = {}
    for key, canon in _NAME_KEYS.items():
        at = q.find(f" {key} ")
        if at >= 0:
            first_seen[canon] = min(at, first_seen.get(canon, at))
    return sorted(first_seen, key=first_seen.__getitem__)  # order of mention


def resolve_city(name: str) -> tuple | None:
    """(city, country, region, lat, lon) for a known city, else the geocoder's
    first hit (so a tool can answer for Pune even though it isn't in CITIES)."""
    known = _NAME_KEYS.get(norm(name).strip())
    if known:
        return _BY_NAME[known]
    hit = geocode(name)
    return (hit[0], hit[1], hit[2] or "elsewhere", hit[3], hit[4]) if hit else None
