"""Tools the model can call when retrieved notes aren't enough: ranking cities,
arbitrary dates, and travel advice. Every number is computed here in code and
handed back as JSON -- the model explains results, it never does the arithmetic."""

import json
import time
from datetime import date, timedelta

import climate
import rag
from data import CITIES, RAIN_MM, _get
from places import REGION_KEYWORDS, norm, resolve_city

REGION_OF = {c[0]: c[2] for c in CITIES}

DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"
ARCHIVE_LAG_DAYS = 7  # archive data is final ~a week behind; newer days come from the forecast API
MAX_RANGE_DAYS = 366
MAX_FORECAST_AHEAD = 15
RANGE_CACHE_SETTLED_S = 24 * 3600
RANGE_CACHE_LIVE_S = 30 * 60
RANGE_CACHE_MAX = 256
MAX_CITY_CHARS = 80
EARLIEST_DATE = date(1940, 1, 1)  # start of the ERA5 archive
METRICS = ("rain_days", "rain_mm", "avg_high_c", "avg_low_c")
WINDOWS = ("next_week", "past_week", "year_ago_week")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rank_cities",
            "description": (
                "Rank the ~100 tracked cities by a weather metric over a window, optionally within one "
                "region. Use for 'rainiest / hottest / coldest cities' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["rain_days", "rain_mm", "avg_high_c", "avg_low_c"]},
                    "window": {"type": "string", "enum": ["next_week", "past_week", "year_ago_week"]},
                    "region": {
                        "type": "string",
                        "description": "Asia, Europe, North America, South America, Africa or Oceania (optional)",
                    },
                    "order": {"type": "string", "enum": ["highest", "lowest"]},
                    "limit": {"type": "integer", "description": "default 5, max 15"},
                },
                "required": ["metric", "window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_range",
            "description": (
                "Daily weather for ANY city (even untracked ones) over a date range: past years from the "
                "archive, or up to 15 days ahead. Returns totals, rain-day count, extremes. Use for "
                "specific dates/periods, e.g. 'last Christmas', 'March 2025', 'next 10 days'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD (max 366 days after start)"},
                },
                "required": ["city", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "travel_advice",
            "description": (
                "Is now (or a given month) a good time to visit a city? Returns 5-year monthly climate "
                "normals, a comfort score per month, the best months, and how the coming week's forecast "
                "compares with normal. Works for any city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "month": {"type": "string", "description": "optional month name, default = current month"},
                },
                "required": ["city"],
            },
        },
    },
]


def _window_slice(window: str, raw: dict) -> dict | None:
    f, a = raw["forecast"]["daily"], raw["archive"]
    if window == "past_week":
        return {k: v[:7] for k, v in f.items()}
    if window == "next_week":
        return {k: v[7:14] for k, v in f.items()}
    return a["daily"] if a else None


def _stats(daily: dict) -> dict:
    mm = [x or 0 for x in daily["precipitation_sum"]]
    highs = [x for x in daily["temperature_2m_max"] if x is not None]
    lows = [x for x in daily["temperature_2m_min"] if x is not None]
    return {
        "days": len(mm),
        "rain_days": sum(x >= RAIN_MM for x in mm),
        "rain_mm": round(sum(mm), 1),
        "avg_high_c": round(sum(highs) / len(highs), 1) if highs else None,
        "avg_low_c": round(sum(lows) / len(lows), 1) if lows else None,
    }


def _valid_city(city) -> str | None:
    """Returns a problem description, or None if the name is usable."""
    if not isinstance(city, str) or not city.strip():
        return "city must be a non-empty string"
    if len(city) > MAX_CITY_CHARS:
        return f"city name too long (max {MAX_CITY_CHARS} characters)"
    return None


def rank_cities(metric: str, window: str, region: str | None = None, order: str = "highest", limit: int = 5) -> dict:
    if metric not in METRICS:
        return {"error": f"metric must be one of {list(METRICS)}"}
    if window not in WINDOWS:
        return {"error": f"window must be one of {list(WINDOWS)}"}
    if order not in ("highest", "lowest"):
        return {"error": "order must be highest or lowest"}
    limit = max(1, min(int(limit), 15))  # a negative limit would slice from the end
    wanted = None
    if region:
        wanted = REGION_KEYWORDS.get(norm(str(region)[:40]).strip()) or str(region)[:40].title()
    rows = []
    for city, raw in rag.snapshot().raw.items():
        if wanted and REGION_OF[city] != wanted:
            continue
        daily = _window_slice(window, raw)
        if daily:
            rows.append({"city": city, "region": REGION_OF[city], **_stats(daily)})
    # Ties (e.g. many cities at 7 rain days) break on total rainfall, same direction.
    rows.sort(key=lambda r: (r[metric] if r[metric] is not None else -1e9, r["rain_mm"]), reverse=(order == "highest"))
    return {
        "metric": metric, "window": window, "region": wanted or "all", "order": order,
        "rain_day_definition": f">= {RAIN_MM:g} mm",
        "results": [{k: r[k] for k in ("city", "region", metric, "days")} for r in rows[:limit]],
        "cities_considered": len(rows),
    }


_range_cache: dict = {}  # (lat, lon, start, end) -> (stored_at, daily)


def _fetch_range(meta: tuple, start: date, end: date) -> dict:
    """Archive for the settled past, forecast API for the last week and ahead,
    merged. Cached so repeated questions don't spend Open-Meteo quota: settled
    history is final (a day), anything touching recent/future days goes stale (30 min)."""
    cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    key = (meta[3], meta[4], start, end)
    ttl = RANGE_CACHE_SETTLED_S if end <= cutoff else RANGE_CACHE_LIVE_S
    hit = _range_cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]

    base = {"latitude": meta[3], "longitude": meta[4], "timezone": "auto", "daily": DAILY}
    parts = []
    if start <= cutoff:
        parts.append(_get(
            "https://archive-api.open-meteo.com/v1/archive",
            {**base, "start_date": start.isoformat(), "end_date": min(end, cutoff).isoformat()},
            on_demand=True,
        )[0]["daily"])
    if end > cutoff:
        parts.append(_get(
            "https://api.open-meteo.com/v1/forecast",
            {**base, "start_date": max(start, cutoff + timedelta(days=1)).isoformat(), "end_date": end.isoformat()},
            on_demand=True,
        )[0]["daily"])
    daily = {k: sum((p[k] for p in parts), []) for k in parts[0]}

    if len(_range_cache) >= RANGE_CACHE_MAX:
        _range_cache.pop(next(iter(_range_cache)))  # evict the oldest entry
    _range_cache[key] = (time.monotonic(), daily)
    return daily


def get_weather_range(city: str, start_date: str, end_date: str) -> dict:
    if problem := _valid_city(city):
        return {"error": problem}
    meta = resolve_city(city)
    if not meta:
        return {"error": f"could not find a city called {city!r}"}
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if end < start:
        return {"error": "end_date is before start_date"}
    if start < EARLIEST_DATE:
        return {"error": f"no data before {EARLIEST_DATE}"}
    if (end - start).days >= MAX_RANGE_DAYS:
        return {"error": f"range too long (max {MAX_RANGE_DAYS} days)"}
    if end > date.today() + timedelta(days=MAX_FORECAST_AHEAD):
        return {"error": f"forecasts only reach {MAX_FORECAST_AHEAD} days ahead"}
    daily = _fetch_range(meta, start, end)
    out = {"city": f"{meta[0]}, {meta[1]}", "start": start_date, "end": end_date, **_stats(daily)}
    out["rain_day_definition"] = f">= {RAIN_MM:g} mm"
    if out["days"] <= 14:
        out["daily"] = [
            {"date": t, "high_c": hi, "low_c": lo, "rain_mm": mm}
            for t, hi, lo, mm in zip(
                daily["time"],
                daily["temperature_2m_max"],
                daily["temperature_2m_min"],
                daily["precipitation_sum"],
                strict=True,
            )
        ]
    return out


def travel_advice(city: str, month: str | None = None) -> dict:
    if problem := _valid_city(city):
        return {"error": problem}
    meta = resolve_city(city)
    if not meta:
        return {"error": f"could not find a city called {city!r}"}
    table = climate.normals(meta[3], meta[4])
    for m in table:
        m["comfort_score"] = climate.comfort_score(m)

    target = date.today().month - 1
    if month:
        prefix = str(month).strip()[:3].title()
        target = next((i for i, m in enumerate(table) if m["month"] == prefix), target)
    ranked = sorted(range(12), key=lambda i: -table[i]["comfort_score"])
    out = {
        "city": f"{meta[0]}, {meta[1]}",
        "scoring": (
            f"comfort_score 0-100 = 100 - 4 per degC the average high is outside "
            f"{climate.COMFORT_LOW:g}-{climate.COMFORT_HIGH:g}C - 2.5 per rain day (>= {RAIN_MM:g} mm). "
            f"Normals are {climate.YEARS}-year monthly averages. Crude heuristic: ignores humidity, crowds, prices."
        ),
        "month_asked": {**table[target], "rank_of_12": ranked.index(target) + 1},
        "best_months": [table[i] for i in ranked[:3]],
        "worst_month": table[ranked[-1]],
    }
    if target == date.today().month - 1:  # "now": compare the coming week with normal
        week = _fetch_range(meta, date.today(), date.today() + timedelta(days=6))
        s = _stats(week)
        normal = table[target]
        out["next_7_days_forecast"] = {
            **s,
            "normal_rain_days_per_7": round(normal["rain_days"] * 7 / 30, 1),
            "normal_avg_high_c": normal["avg_high_c"],
        }
    return out


_DISPATCH = {"rank_cities": rank_cities, "get_weather_range": get_weather_range, "travel_advice": travel_advice}


def run(name: str, arguments: str) -> str:
    """Executes a tool call; errors go back to the model as data so it can retry or explain."""
    try:
        fn = _DISPATCH[name]
        return json.dumps(fn(**json.loads(arguments or "{}")))
    except Exception as err:  # bad args, network failure, unknown tool...
        return json.dumps({"error": f"{type(err).__name__}: {err}"})
