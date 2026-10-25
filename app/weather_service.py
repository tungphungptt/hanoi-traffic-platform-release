from __future__ import annotations

import copy
import json
import math
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from app.hanoi_scope import is_hanoi_coordinate

WEATHER_SOURCE = "Open-Meteo"
WEATHER_SOURCE_URL = "https://open-meteo.com/"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 600
STALE_MAX_SECONDS = 3600
CACHE_MAX_ENTRIES = 128
_CACHE: dict[tuple[float, float], tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 56: "Freezing drizzle", 57: "Dense freezing drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow",
    73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Heavy rain showers",
    85: "Light snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

class WeatherServiceError(RuntimeError):
    pass


def _finite(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} is out of range")
    return number


def weather_description(code):
    try:
        key = int(code)
    except (TypeError, ValueError):
        return "Unknown weather condition"
    return WMO_CODES.get(key, "Unknown weather condition")


def normalize_weather_payload(payload, latitude, longitude, *, fetched_at=None, is_stale=False, stale_age_seconds=0):
    if not isinstance(payload, dict) or not isinstance(payload.get("current"), dict):
        raise WeatherServiceError("Weather provider returned an invalid current-weather object")
    current = payload["current"]
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), dict) else {}

    def number_or_none(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            n = float(value)
            return round(n, 2) if math.isfinite(n) else None
        except (TypeError, ValueError):
            return None

    current_code = current.get("weather_code")
    times = hourly.get("time") if isinstance(hourly.get("time"), list) else []
    values = {}
    for key in ("temperature_2m", "precipitation_probability", "precipitation", "weather_code", "wind_speed_10m"):
        raw = hourly.get(key)
        values[key] = raw if isinstance(raw, list) else []

    current_time = str(current.get("time") or "")
    start_idx = next((i for i, stamp in enumerate(times) if str(stamp) >= current_time), 0)
    hourly_rows = []
    for i in range(start_idx, min(len(times), start_idx + 7)):
        code = values["weather_code"][i] if i < len(values["weather_code"]) else None
        hourly_rows.append({
            "time": str(times[i]),
            "temperature_c": number_or_none(values["temperature_2m"][i] if i < len(values["temperature_2m"]) else None),
            "precipitation_probability_pct": number_or_none(values["precipitation_probability"][i] if i < len(values["precipitation_probability"]) else None),
            "precipitation_mm": number_or_none(values["precipitation"][i] if i < len(values["precipitation"]) else None),
            "wind_speed_kmh": number_or_none(values["wind_speed_10m"][i] if i < len(values["wind_speed_10m"]) else None),
            "weather_code": code,
            "description": weather_description(code),
        })

    return {
        "ok": True,
        "source": WEATHER_SOURCE,
        "source_url": WEATHER_SOURCE_URL,
        "attribution": "Weather data by Open-Meteo (CC BY 4.0)",
        "location": {
            "latitude": round(float(latitude), 5),
            "longitude": round(float(longitude), 5),
            "timezone": str(payload.get("timezone") or "Asia/Bangkok"),
        },
        "current": {
            "time": current_time,
            "temperature_c": number_or_none(current.get("temperature_2m")),
            "apparent_temperature_c": number_or_none(current.get("apparent_temperature")),
            "relative_humidity_pct": number_or_none(current.get("relative_humidity_2m")),
            "wind_speed_kmh": number_or_none(current.get("wind_speed_10m")),
            "wind_direction_deg": number_or_none(current.get("wind_direction_10m")),
            "precipitation_mm": number_or_none(current.get("precipitation")),
            "rain_mm": number_or_none(current.get("rain")),
            "weather_code": current_code,
            "description": weather_description(current_code),
        },
        "hourly": hourly_rows,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_stale": bool(is_stale),
        "stale_age_seconds": int(max(0, stale_age_seconds)) if is_stale else 0,
    }


def get_weather_report(latitude=21.0285, longitude=105.8542, *, timeout=8):
    lat = _finite(latitude, "latitude", -90, 90)
    lon = _finite(longitude, "longitude", -180, 180)
    if not is_hanoi_coordinate(lat, lon):
        raise ValueError("outside_hanoi_boundary: weather is only available within Hanoi.")
    key = (round(lat, 3), round(lon, 3))
    now_mono = time.monotonic()
    cached = None
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry:
            cached_at, cached_data = entry
            age = max(0, now_mono - cached_at)
            cached = (age, copy.deepcopy(cached_data))
            if age <= CACHE_TTL_SECONDS:
                return cached[1]

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m",
        "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m",
        "forecast_days": 2,
        "timezone": "Asia/Bangkok",
    }
    request = Request(
        WEATHER_API_URL + "?" + urlencode(params),
        headers={"User-Agent": "HanoiTrafficPlatform/1.0 (weather dashboard)", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=max(2, min(float(timeout), 12))) as response:
            if response.status != 200:
                raise WeatherServiceError(f"Open-Meteo returned HTTP {response.status}")
            payload = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
        result = normalize_weather_payload(payload, lat, lon)
        with _CACHE_LOCK:
            if len(_CACHE) >= CACHE_MAX_ENTRIES:
                oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
                _CACHE.pop(oldest, None)
            _CACHE[key] = (time.monotonic(), copy.deepcopy(result))
        return result
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, WeatherServiceError) as exc:
        if cached and cached[0] <= STALE_MAX_SECONDS:
            stale = cached[1]
            stale["is_stale"] = True
            stale["stale_age_seconds"] = int(cached[0])
            stale["warning"] = "Upstream weather unavailable; showing explicitly marked cached data."
            return stale
        if isinstance(exc, WeatherServiceError):
            raise
        raise WeatherServiceError("Open-Meteo is unavailable or returned invalid data") from exc
