"""Weather from Open-Meteo (https://open-meteo.com, free, no API key).

Per city we build three retrievable notes:
  forecast  - current conditions + today and the next 6 days
  past_week - the 7 days before today
  year_ago  - the 7 days centred on today's date one year ago (archive API)

Rain-day counts are computed here, in code, and written into the note: LLMs
are unreliable at counting, so the model reads the answer instead of deriving it."""

import functools
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

# (city, country, region, lat, lon)
CITIES = [
    # Asia
    ("Mumbai", "India", "Asia", 19.076, 72.8777),
    ("Delhi", "India", "Asia", 28.6139, 77.209),
    ("Bangalore", "India", "Asia", 12.9716, 77.5946),
    ("Chennai", "India", "Asia", 13.0827, 80.2707),
    ("Kolkata", "India", "Asia", 22.5726, 88.3639),
    ("Hyderabad", "India", "Asia", 17.385, 78.4867),
    ("Colombo", "Sri Lanka", "Asia", 6.9271, 79.8612),
    ("Dhaka", "Bangladesh", "Asia", 23.8103, 90.4125),
    ("Karachi", "Pakistan", "Asia", 24.8607, 67.0011),
    ("Dubai", "UAE", "Asia", 25.2048, 55.2708),
    ("Riyadh", "Saudi Arabia", "Asia", 24.7136, 46.6753),
    ("Tehran", "Iran", "Asia", 35.6892, 51.389),
    ("Tokyo", "Japan", "Asia", 35.6762, 139.6503),
    ("Seoul", "South Korea", "Asia", 37.5665, 126.978),
    ("Beijing", "China", "Asia", 39.9042, 116.4074),
    ("Shanghai", "China", "Asia", 31.2304, 121.4737),
    ("Hong Kong", "China", "Asia", 22.3193, 114.1694),
    ("Singapore", "Singapore", "Asia", 1.3521, 103.8198),
    ("Bangkok", "Thailand", "Asia", 13.7563, 100.5018),
    ("Kuala Lumpur", "Malaysia", "Asia", 3.139, 101.6869),
    ("Jakarta", "Indonesia", "Asia", -6.2088, 106.8456),
    ("Manila", "Philippines", "Asia", 14.5995, 120.9842),
    ("Ho Chi Minh City", "Vietnam", "Asia", 10.8231, 106.6297),
    # Europe
    ("London", "UK", "Europe", 51.5074, -0.1278),
    ("Paris", "France", "Europe", 48.8566, 2.3522),
    ("Berlin", "Germany", "Europe", 52.52, 13.405),
    ("Madrid", "Spain", "Europe", 40.4168, -3.7038),
    ("Rome", "Italy", "Europe", 41.9028, 12.4964),
    ("Amsterdam", "Netherlands", "Europe", 52.3676, 4.9041),
    ("Vienna", "Austria", "Europe", 48.2082, 16.3738),
    ("Prague", "Czechia", "Europe", 50.0755, 14.4378),
    ("Budapest", "Hungary", "Europe", 47.4979, 19.0402),
    ("Warsaw", "Poland", "Europe", 52.2297, 21.0122),
    ("Stockholm", "Sweden", "Europe", 59.3293, 18.0686),
    ("Oslo", "Norway", "Europe", 59.9139, 10.7522),
    ("Copenhagen", "Denmark", "Europe", 55.6761, 12.5683),
    ("Helsinki", "Finland", "Europe", 60.1699, 24.9384),
    ("Dublin", "Ireland", "Europe", 53.3498, -6.2603),
    ("Lisbon", "Portugal", "Europe", 38.7223, -9.1393),
    ("Athens", "Greece", "Europe", 37.9838, 23.7275),
    ("Istanbul", "Turkey", "Europe", 41.0082, 28.9784),
    ("Kyiv", "Ukraine", "Europe", 50.4501, 30.5234),
    ("Moscow", "Russia", "Europe", 55.7558, 37.6173),
    # North America
    ("New York", "USA", "North America", 40.7128, -74.006),
    ("Los Angeles", "USA", "North America", 34.0522, -118.2437),
    ("Chicago", "USA", "North America", 41.8781, -87.6298),
    ("Houston", "USA", "North America", 29.7604, -95.3698),
    ("Phoenix", "USA", "North America", 33.4484, -112.074),
    ("San Francisco", "USA", "North America", 37.7749, -122.4194),
    ("Seattle", "USA", "North America", 47.6062, -122.3321),
    ("Miami", "USA", "North America", 25.7617, -80.1918),
    ("Dallas", "USA", "North America", 32.7767, -96.797),
    ("Boston", "USA", "North America", 42.3601, -71.0589),
    ("Denver", "USA", "North America", 39.7392, -104.9903),
    ("Atlanta", "USA", "North America", 33.749, -84.388),
    ("Washington DC", "USA", "North America", 38.9072, -77.0369),
    ("Toronto", "Canada", "North America", 43.6532, -79.3832),
    ("Vancouver", "Canada", "North America", 49.2827, -123.1207),
    ("Montreal", "Canada", "North America", 45.5017, -73.5673),
    ("Mexico City", "Mexico", "North America", 19.4326, -99.1332),
    ("Havana", "Cuba", "North America", 23.1136, -82.3666),
    # South America
    ("Sao Paulo", "Brazil", "South America", -23.5505, -46.6333),
    ("Rio de Janeiro", "Brazil", "South America", -22.9068, -43.1729),
    ("Brasilia", "Brazil", "South America", -15.7939, -47.8828),
    ("Buenos Aires", "Argentina", "South America", -34.6037, -58.3816),
    ("Santiago", "Chile", "South America", -33.4489, -70.6693),
    ("Lima", "Peru", "South America", -12.0464, -77.0428),
    ("Bogota", "Colombia", "South America", 4.711, -74.0721),
    ("Caracas", "Venezuela", "South America", 10.4806, -66.9036),
    ("Quito", "Ecuador", "South America", -0.1807, -78.4678),
    ("Montevideo", "Uruguay", "South America", -34.9011, -56.1645),
    # Africa
    ("Cairo", "Egypt", "Africa", 30.0444, 31.2357),
    ("Casablanca", "Morocco", "Africa", 33.5731, -7.5898),
    ("Algiers", "Algeria", "Africa", 36.7538, 3.0588),
    ("Tunis", "Tunisia", "Africa", 36.8065, 10.1815),
    ("Dakar", "Senegal", "Africa", 14.7167, -17.4677),
    ("Accra", "Ghana", "Africa", 5.6037, -0.187),
    ("Lagos", "Nigeria", "Africa", 6.5244, 3.3792),
    ("Kinshasa", "DR Congo", "Africa", -4.4419, 15.2663),
    ("Luanda", "Angola", "Africa", -8.839, 13.2894),
    ("Khartoum", "Sudan", "Africa", 15.5007, 32.5599),
    ("Addis Ababa", "Ethiopia", "Africa", 9.03, 38.74),
    ("Nairobi", "Kenya", "Africa", -1.2921, 36.8219),
    ("Kampala", "Uganda", "Africa", 0.3476, 32.5825),
    ("Dar es Salaam", "Tanzania", "Africa", -6.7924, 39.2083),
    ("Johannesburg", "South Africa", "Africa", -26.2041, 28.0473),
    ("Cape Town", "South Africa", "Africa", -33.9249, 18.4241),
    # Oceania
    ("Sydney", "Australia", "Oceania", -33.8688, 151.2093),
    ("Melbourne", "Australia", "Oceania", -37.8136, 144.9631),
    ("Brisbane", "Australia", "Oceania", -27.4698, 153.0251),
    ("Perth", "Australia", "Oceania", -31.9505, 115.8605),
    ("Adelaide", "Australia", "Oceania", -34.9285, 138.6007),
    ("Canberra", "Australia", "Oceania", -35.2809, 149.13),
    ("Darwin", "Australia", "Oceania", -12.4634, 130.8456),
    ("Hobart", "Australia", "Oceania", -42.8821, 147.3272),
    ("Auckland", "New Zealand", "Oceania", -36.8485, 174.7633),
    ("Wellington", "New Zealand", "Oceania", -41.2866, 174.7756),
]

# WMO weather interpretation codes -> plain English.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

RAIN_MM = 1.0  # a day counts as a "rain day" at or above this much precipitation
CHUNK = 25  # cities per Open-Meteo request (it accepts comma-separated coordinates)
DAILY_VARS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"


@dataclass
class WeatherDoc:
    city: str
    region: str
    kind: str  # "forecast" | "past_week" | "year_ago"
    text: str


# --- Being a good citizen of the free Open-Meteo API (limits: 600 calls/min, 10,000/day) ---
MIN_INTERVAL_S = 0.25  # at most ~4 outbound calls/s, ever, whoever is asking
ON_DEMAND_PER_DAY = int(os.environ.get("UPSTREAM_ON_DEMAND_PER_DAY", "500"))
_gate = threading.Lock()
_last_call = 0.0
_budget_day = date.today()
_budget_used = 0


class UpstreamBudgetError(RuntimeError):
    """Today's allowance of user-triggered Open-Meteo calls is spent."""


def _spend_budget() -> None:
    """User-triggered calls (tools, geocoding, climate) share a daily allowance, so
    chat traffic can never eat the quota the hourly refresh needs. The refresh
    itself is ~8 calls/hour and is not counted."""
    global _budget_day, _budget_used
    with _gate:
        if _budget_day != date.today():
            _budget_day, _budget_used = date.today(), 0
        if _budget_used >= ON_DEMAND_PER_DAY:
            raise UpstreamBudgetError(
                f"daily limit of {ON_DEMAND_PER_DAY} live weather lookups reached; try again tomorrow"
            )
        _budget_used += 1


def _wait_turn() -> None:
    global _last_call
    with _gate:  # held while sleeping on purpose: callers queue up one by one
        wait = _last_call + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _get(url: str, params: dict, on_demand: bool = False) -> list[dict]:
    """Throttled GET with backoff on 429 / network errors, honouring Retry-After.
    Always returns a list -- the API returns a bare object for a single location."""
    if on_demand:
        _spend_budget()
    for attempt in range(4):
        _wait_turn()
        try:
            res = httpx.get(url, params=params, timeout=30)
        except httpx.TransportError:  # timeout / dropped connection
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
            continue
        if res.status_code == 429 and attempt < 3:
            try:
                delay = float(res.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2 ** (attempt + 1)
            time.sleep(min(delay, 30))
            continue
        res.raise_for_status()
        data = res.json()
        return data if isinstance(data, list) else [data]
    raise RuntimeError("unreachable")

def _sky(code) -> str:
    return WEATHER_CODES.get(code, f"weather code {code}")


def _day_line(daily: dict, i: int, with_prob: bool) -> str:
    d = date.fromisoformat(daily["time"][i])
    mm = daily["precipitation_sum"][i] or 0
    line = (
        f"{d:%a %Y-%m-%d} {_sky(daily['weather_code'][i])}, "
        f"{daily['temperature_2m_min'][i]}-{daily['temperature_2m_max'][i]}C, {mm} mm rain"
    )
    probs = daily.get("precipitation_probability_max") if with_prob else None
    if probs and probs[i] is not None:
        line += f" ({probs[i]}% chance)"
    return line


def _rain_summary(daily: dict, idx: range, tense: str) -> str:
    rainy = [i for i in idx if (daily["precipitation_sum"][i] or 0) >= RAIN_MM]
    total = round(sum(daily["precipitation_sum"][i] or 0 for i in idx), 1)
    days = ", ".join(f"{date.fromisoformat(daily['time'][i]):%a %d %b}" for i in rainy) or "none"
    return (
        f"RAIN DAYS ({tense}, >= {RAIN_MM:g} mm): {len(rainy)} of {len(idx)} days [{days}]; "
        f"total precipitation {total} mm."
    )


def _head(c: tuple) -> str:
    return f"{c[0]}, {c[1]} ({c[2]})"


def _forecast_doc(c: tuple, d: dict) -> WeatherDoc:
    cur, daily = d["current"], d["daily"]
    today = range(7, 14)  # past_days=7 puts today at index 7
    text = (
        f"{_head(c)} - FORECAST issued {cur['time'][:10]}. Now: {_sky(cur['weather_code'])}, "
        f"{cur['temperature_2m']}C (feels {cur['apparent_temperature']}C), "
        f"{cur['relative_humidity_2m']}% humidity, wind {cur['wind_speed_10m']} km/h. "
        f"Today and the next 6 days: "
        + "; ".join(_day_line(daily, i, True) for i in today)
        + ". "
        + _rain_summary(daily, today, "next 7 days incl. today")
    )
    return WeatherDoc(c[0], c[2], "forecast", text)


def _past_doc(c: tuple, d: dict) -> WeatherDoc:
    daily = d["daily"]
    week = range(0, 7)
    text = (
        f"{_head(c)} - PAST 7 DAYS ({daily['time'][0]} to {daily['time'][6]}): "
        + "; ".join(_day_line(daily, i, False) for i in week)
        + ". "
        + _rain_summary(daily, week, "past 7 days")
    )
    return WeatherDoc(c[0], c[2], "past_week", text)


def _year_ago_doc(c: tuple, d: dict) -> WeatherDoc:
    daily = d["daily"]
    n = len(daily["time"])
    target = daily["time"][n // 2]
    text = (
        f"{_head(c)} - ONE YEAR AGO (the week around {target}, historical archive): "
        + "; ".join(_day_line(daily, i, False) for i in range(n))
        + ". "
        + _rain_summary(daily, range(n), "that week")
    )
    return WeatherDoc(c[0], c[2], "year_ago", text)


def _one_year_ago(today: date) -> date:
    try:
        return today.replace(year=today.year - 1)
    except ValueError:  # Feb 29
        return today.replace(year=today.year - 1, day=28)


@dataclass
class Snapshot:
    """Everything fetched in one refresh: the retrievable notes plus the raw
    daily series, which the analytics tools (tools.py) compute over."""
    docs: list[WeatherDoc]
    raw: dict  # city -> {"forecast": open-meteo json, "archive": json | None}


CACHE_FILE = Path(__file__).parent / ".cache" / "snapshot.json"
SNAPSHOT_TTL_S = 60 * 60


def _fetch_raw() -> dict:
    target = _one_year_ago(date.today())
    raw: dict = {}
    for start in range(0, len(CITIES), CHUNK):
        chunk = CITIES[start:start + CHUNK]
        coords = {
            "latitude": ",".join(str(c[3]) for c in chunk),
            "longitude": ",".join(str(c[4]) for c in chunk),
            "timezone": "auto",
        }
        forecast = _get(
            "https://api.open-meteo.com/v1/forecast",
            {
                **coords,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": DAILY_VARS + ",precipitation_probability_max",
                "past_days": 7,
                "forecast_days": 7,
            },
        )
        try:
            archive = _get(
                "https://archive-api.open-meteo.com/v1/archive",
                {
                    **coords,
                    "daily": DAILY_VARS,
                    "start_date": (target - timedelta(days=3)).isoformat(),
                    "end_date": (target + timedelta(days=3)).isoformat(),
                },
            )
        except httpx.HTTPError as err:  # history is a bonus; don't block startup on it
            print(f"archive fetch failed for cities {start}-{start + len(chunk) - 1}: {err}")
            archive = [None] * len(chunk)
        for c, f, a in zip(chunk, forecast, archive, strict=True):
            raw[c[0]] = {"forecast": f, "archive": a}
    return raw


def _build_docs(raw: dict) -> list[WeatherDoc]:
    docs: list[WeatherDoc] = []
    for c in CITIES:
        r = raw.get(c[0])
        if not r:
            continue
        docs.append(_forecast_doc(c, r["forecast"]))
        docs.append(_past_doc(c, r["forecast"]))
        if r["archive"] is not None:
            docs.append(_year_ago_doc(c, r["archive"]))
    return docs


def load_snapshot(force: bool = False) -> Snapshot:
    """Reads the on-disk cache when it is under an hour old and from today
    (restarts are then instant); otherwise fetches fresh and rewrites it."""
    if not force and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            fresh = time.time() - cached["fetched_at"] < SNAPSHOT_TTL_S
            if fresh and cached["date"] == date.today().isoformat():
                print("Using cached weather snapshot.")
                return Snapshot(_build_docs(cached["raw"]), cached["raw"])
        except (OSError, ValueError, KeyError):
            pass  # corrupt cache -> just refetch
    raw = _fetch_raw()
    now = time.time()
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps({"fetched_at": now, "date": date.today().isoformat(), "raw": raw}), encoding="utf-8"
        )
    except OSError as err:
        print(f"could not write weather cache: {err}")
    return Snapshot(_build_docs(raw), raw)


@functools.lru_cache(maxsize=512)  # place names don't move; repeat questions cost nothing
def geocode(name: str) -> tuple | None:
    """Finds a city that isn't in CITIES via Open-Meteo's free geocoder."""
    body = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": name, "count": 1}, on_demand=True)[0]
    hits = body.get("results") or []
    if not hits:
        return None
    h = hits[0]
    return (h["name"], h.get("country", ""), h.get("admin1", ""), h["latitude"], h["longitude"])
