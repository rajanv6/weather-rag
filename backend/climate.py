"""Climate normals for travel questions: five full years of daily archive data
reduced to per-month averages. Fetched lazily, one city at a time (a single
archive call, ~1s), and cached on disk -- no need to preload every city."""

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from data import RAIN_MM, _get

CACHE_FILE = Path(__file__).parent / ".cache" / "climate.json"
YEARS = 5
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COMFORT_LOW, COMFORT_HIGH = 20.0, 28.0  # comfortable daytime high, C


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def normals(lat: float, lon: float) -> list[dict]:
    """Per-month averages (index 0 = January): avg daily high/low in C, rain
    days per month (>= RAIN_MM) and rain mm per month."""
    key = f"{lat:.2f},{lon:.2f}"
    cache = _load_cache()
    if key in cache:
        return cache[key]

    last = date.today().year - 1
    daily = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "start_date": f"{last - YEARS + 1}-01-01", "end_date": f"{last}-12-31",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        },
        on_demand=True,
    )[0]["daily"]

    hi, lo = defaultdict(list), defaultdict(list)
    rain_days, rain_mm = defaultdict(int), defaultdict(float)
    for t, tmax, tmin, mm in zip(
        daily["time"], daily["temperature_2m_max"], daily["temperature_2m_min"], daily["precipitation_sum"],
        strict=True,
    ):
        if tmax is None or tmin is None:
            continue
        m = int(t[5:7])
        hi[m].append(tmax)
        lo[m].append(tmin)
        rain_days[m] += (mm or 0) >= RAIN_MM
        rain_mm[m] += mm or 0

    result = [
        {
            "month": MONTHS[m - 1],
            "avg_high_c": round(sum(hi[m]) / len(hi[m]), 1),
            "avg_low_c": round(sum(lo[m]) / len(lo[m]), 1),
            "rain_days": round(rain_days[m] / YEARS, 1),
            "rain_mm": round(rain_mm[m] / YEARS),
        }
        for m in range(1, 13)
    ]
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache | {key: result}), encoding="utf-8")
    except OSError as err:
        print(f"could not write climate cache: {err}")
    return result


def comfort_score(month: dict) -> int:
    """0-100 sightseeing heuristic: 4 points off per degree the typical high is
    outside 20-28C, 2.5 points off per rain day. Crude on purpose -- the tool
    output states the formula so the model can explain it."""
    high = month["avg_high_c"]
    temp_penalty = 4 * max(0.0, COMFORT_LOW - high, high - COMFORT_HIGH)
    return round(max(0.0, 100 - temp_penalty - 2.5 * month["rain_days"]))
