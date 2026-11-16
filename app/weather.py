"""Weather now and forecast for points in Hanoi, from Open-Meteo (CC BY 4.0), cached 10 minutes."""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.hanoi_scope import is_hanoi_coordinate

API_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Dữ liệu thời tiết: Open-Meteo (CC BY 4.0)"
CACHE_SECONDS = 600
STALE_SECONDS = 3 * 3600
TIMEZONE = "Asia/Bangkok"  # UTC+7, same clock as Hanoi

# One representative point per area, so the map shows the whole city at a glance.
AREAS: list[tuple[str, float, float]] = [
    ("Hoàn Kiếm", 21.0285, 105.8542), ("Cầu Giấy", 21.0362, 105.7906), ("Long Biên", 21.0465, 105.8885),
    ("Hoàng Mai", 20.9745, 105.8628), ("Hà Đông", 20.9714, 105.7788), ("Bắc Từ Liêm", 21.0700, 105.7500),
    ("Đông Anh", 21.1370, 105.8460), ("Gia Lâm", 21.0213, 105.9470), ("Mê Linh", 21.1830, 105.7170),
    ("Sóc Sơn", 21.2560, 105.8480), ("Sơn Tây", 21.1380, 105.5050), ("Ba Vì", 21.1990, 105.4230),
    ("Quốc Oai", 20.9900, 105.6400), ("Chương Mỹ", 20.9200, 105.7050), ("Thường Tín", 20.8720, 105.8600),
    ("Phú Xuyên", 20.7400, 105.9100), ("Mỹ Đức", 20.6830, 105.7440),
]

WMO_VI = {
    0: "Trời quang", 1: "Ít mây", 2: "Mây rải rác", 3: "Nhiều mây", 45: "Sương mù", 48: "Sương mù đóng băng",
    51: "Mưa phùn nhẹ", 53: "Mưa phùn", 55: "Mưa phùn dày", 56: "Mưa phùn lạnh", 57: "Mưa phùn lạnh dày",
    61: "Mưa nhỏ", 63: "Mưa vừa", 65: "Mưa to", 66: "Mưa lạnh", 67: "Mưa lạnh to",
    71: "Tuyết nhẹ", 73: "Tuyết", 75: "Tuyết dày", 77: "Hạt tuyết",
    80: "Mưa rào nhẹ", 81: "Mưa rào", 82: "Mưa rào rất to", 85: "Mưa tuyết nhẹ", 86: "Mưa tuyết",
    95: "Dông", 96: "Dông kèm mưa đá", 99: "Dông kèm mưa đá lớn",
}

CURRENT = "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m"
HOURLY = "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m"
DAILY = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"


class WeatherError(RuntimeError):
    pass


def describe(code: Any) -> str:
    try:
        return WMO_VI.get(int(code), "Không rõ")
    except (TypeError, ValueError):
        return "Không rõ"


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 1) if math.isfinite(number) else None


def _fetch(params: dict[str, Any], timeout: float = 8.0) -> Any:
    request = Request(f"{API_URL}?{urlencode(params)}", headers={"User-Agent": "hanoiFlow/9 (weather)", "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))


class _Cache:
    def __init__(self) -> None:
        self._items: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any, load: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._items.get(key)
        if hit and now - hit[0] <= CACHE_SECONDS:
            return hit[1]
        try:
            value = load()
        except Exception as exc:
            if hit and now - hit[0] <= STALE_SECONDS:
                return {**hit[1], "stale": True, "stale_minutes": int((now - hit[0]) // 60)}
            raise WeatherError("Không lấy được dữ liệu từ Open-Meteo; thử lại sau.") from exc
        with self._lock:
            if len(self._items) > 256:
                self._items.clear()
            self._items[key] = (now, value)
        return value


_cache = _Cache()


def _forecast_payload(raw: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    current = raw.get("current") or {}
    hourly = raw.get("hourly") or {}
    daily = raw.get("daily") or {}
    times = hourly.get("time") or []
    start = next((i for i, t in enumerate(times) if str(t) >= str(current.get("time", ""))[:13]), 0)
    hours = []
    for i in range(start, min(len(times), start + 24)):
        pick = lambda key: (hourly.get(key) or [None] * len(times))[i]  # noqa: E731
        hours.append({"time": times[i], "temperature_c": _num(pick("temperature_2m")),
                      "precipitation_probability_pct": _num(pick("precipitation_probability")),
                      "precipitation_mm": _num(pick("precipitation")), "wind_kmh": _num(pick("wind_speed_10m")),
                      "weather_code": pick("weather_code"), "description": describe(pick("weather_code"))})
    days = []
    for i, day in enumerate(daily.get("time") or []):
        pick = lambda key: (daily.get(key) or [None] * (i + 1))[i]  # noqa: E731
        days.append({"date": day, "max_c": _num(pick("temperature_2m_max")), "min_c": _num(pick("temperature_2m_min")),
                     "precipitation_mm": _num(pick("precipitation_sum")),
                     "precipitation_probability_pct": _num(pick("precipitation_probability_max")),
                     "weather_code": pick("weather_code"), "description": describe(pick("weather_code"))})
    return {
        "location": {"latitude": round(lat, 4), "longitude": round(lon, 4)},
        "current": {"time": current.get("time"), "temperature_c": _num(current.get("temperature_2m")),
                    "apparent_temperature_c": _num(current.get("apparent_temperature")),
                    "humidity_pct": _num(current.get("relative_humidity_2m")),
                    "precipitation_mm": _num(current.get("precipitation")),
                    "wind_kmh": _num(current.get("wind_speed_10m")), "wind_direction_deg": _num(current.get("wind_direction_10m")),
                    "weather_code": current.get("weather_code"), "description": describe(current.get("weather_code"))},
        "hourly": hours,
        "daily": days,
        "attribution": ATTRIBUTION,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def forecast(latitude: float, longitude: float) -> dict[str, Any]:
    """Current weather, next 24 hours and 3 days for one point inside Hanoi."""
    lat, lon = float(latitude), float(longitude)
    if not is_hanoi_coordinate(lat, lon):
        raise ValueError("Điểm này nằm ngoài ranh giới Hà Nội.")
    key = ("point", round(lat, 2), round(lon, 2))  # ~1 km: one upstream call per neighbourhood
    params = {"latitude": lat, "longitude": lon, "current": CURRENT, "hourly": HOURLY, "daily": DAILY,
              "forecast_days": 3, "timezone": TIMEZONE}
    return _cache.get(key, lambda: _forecast_payload(_fetch(params), lat, lon))


def overview() -> dict[str, Any]:
    """Current weather and today's rain chance at one point per area, in a single upstream call."""
    def load() -> dict[str, Any]:
        raw = _fetch({"latitude": ",".join(str(a[1]) for a in AREAS), "longitude": ",".join(str(a[2]) for a in AREAS),
                      "current": CURRENT, "daily": DAILY, "forecast_days": 1, "timezone": TIMEZONE})
        rows = raw if isinstance(raw, list) else [raw]
        if len(rows) != len(AREAS):
            raise WeatherError("Open-Meteo trả về sai số điểm.")
        areas = []
        for (name, lat, lon), item in zip(AREAS, rows):
            current, daily = item.get("current") or {}, item.get("daily") or {}
            first = lambda key: (daily.get(key) or [None])[0]  # noqa: E731
            areas.append({"name": name, "latitude": lat, "longitude": lon,
                          "temperature_c": _num(current.get("temperature_2m")),
                          "precipitation_mm": _num(current.get("precipitation")),
                          "wind_kmh": _num(current.get("wind_speed_10m")),
                          "weather_code": current.get("weather_code"), "description": describe(current.get("weather_code")),
                          "today_max_c": _num(first("temperature_2m_max")), "today_min_c": _num(first("temperature_2m_min")),
                          "today_rain_probability_pct": _num(first("precipitation_probability_max"))})
        return {"areas": areas, "attribution": ATTRIBUTION,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return _cache.get("overview", load)
