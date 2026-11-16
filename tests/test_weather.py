import pytest

from app import weather
from app.hanoi_scope import is_hanoi_coordinate


def test_every_overview_area_is_inside_hanoi():
    assert all(is_hanoi_coordinate(lat, lon) for _, lat, lon in weather.AREAS)
    assert not is_hanoi_coordinate(10.77, 106.70)  # Ho Chi Minh City


def _point_payload():
    hours = [f"2026-11-16T{h:02d}:00" for h in range(24)] + [f"2026-11-17T{h:02d}:00" for h in range(24)]
    return {
        "current": {"time": "2026-11-16T11:45", "temperature_2m": 24.3, "apparent_temperature": 25.0,
                    "relative_humidity_2m": 70, "precipitation": 0.2, "weather_code": 61, "wind_speed_10m": 9.1},
        "hourly": {"time": hours, "temperature_2m": list(range(48)), "precipitation_probability": [30] * 48,
                   "precipitation": [0.1] * 48, "weather_code": [3] * 48, "wind_speed_10m": [8] * 48},
        "daily": {"time": ["2026-11-16", "2026-11-17", "2026-11-18"], "temperature_2m_max": [27, 26, 25],
                  "temperature_2m_min": [20, 19, 18], "precipitation_sum": [2, 0, 0],
                  "precipitation_probability_max": [80, 10, 5], "weather_code": [61, 3, 1]},
    }


def test_point_forecast_starts_at_the_current_hour(monkeypatch):
    monkeypatch.setattr(weather, "_cache", weather._Cache())
    monkeypatch.setattr(weather, "_fetch", lambda params, timeout=8.0: _point_payload())
    result = weather.forecast(21.0285, 105.8542)
    assert result["current"]["description"] == "Mưa nhỏ"
    assert len(result["hourly"]) == 24 and result["hourly"][0]["time"] == "2026-11-16T11:00"
    assert [d["max_c"] for d in result["daily"]] == [27, 26, 25]
    with pytest.raises(ValueError):
        weather.forecast(10.77, 106.70)


def test_stale_cache_is_marked_and_failures_raise(monkeypatch):
    cache = weather._Cache()
    monkeypatch.setattr(weather, "_cache", cache)
    monkeypatch.setattr(weather, "_fetch", lambda params, timeout=8.0: _point_payload())
    weather.forecast(21.0285, 105.8542)
    for key, (stamp, value) in list(cache._items.items()):
        cache._items[key] = (stamp - weather.CACHE_SECONDS - 60, value)

    def down(params, timeout=8.0):
        raise OSError("offline")
    monkeypatch.setattr(weather, "_fetch", down)
    assert weather.forecast(21.0285, 105.8542)["stale"] is True
    with pytest.raises(weather.WeatherError):
        weather.forecast(21.2560, 105.8480)


def test_overview_maps_one_result_per_area(monkeypatch):
    monkeypatch.setattr(weather, "_cache", weather._Cache())
    item = {"current": {"temperature_2m": 22, "weather_code": 0}, "daily": {"temperature_2m_max": [25], "temperature_2m_min": [18],
                                                                           "precipitation_probability_max": [10]}}
    monkeypatch.setattr(weather, "_fetch", lambda params, timeout=8.0: [item] * len(weather.AREAS))
    data = weather.overview()
    assert len(data["areas"]) == len(weather.AREAS) and data["areas"][0]["description"] == "Trời quang"
