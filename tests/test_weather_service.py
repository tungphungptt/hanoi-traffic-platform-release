from app.weather_service import normalize_weather_payload, weather_description, _finite
import pytest


def sample_payload():
    return {
        "timezone": "Asia/Bangkok",
        "current": {
            "time": "2026-10-24T14:00", "temperature_2m": 29.5,
            "apparent_temperature": 32.0, "relative_humidity_2m": 72,
            "wind_speed_10m": 9.0, "wind_direction_10m": 180,
            "precipitation": 0.2, "rain": 0.2, "weather_code": 61,
        },
        "hourly": {
            "time": ["2026-10-24T14:00", "2026-10-24T15:00", "2026-10-24T16:00"],
            "temperature_2m": [29.5, 28.0, 27.0],
            "precipitation_probability": [30, 65, 70],
            "precipitation": [0.2, 1.1, 2.0],
            "weather_code": [61, 63, 65],
            "wind_speed_10m": [9.0, 12.0, 14.0],
        },
    }


def test_weather_payload_is_normalized_without_network():
    result = normalize_weather_payload(sample_payload(), 21.0285, 105.8542, fetched_at="2026-10-24T07:00:00+00:00")
    assert result["ok"] is True
    assert result["source"] == "Open-Meteo"
    assert result["current"]["temperature_c"] == 29.5
    assert len(result["hourly"]) == 3
    assert result["hourly"][1]["precipitation_probability_pct"] == 65.0
    assert result["is_stale"] is False


def test_weather_codes_have_user_facing_description():
    assert weather_description(61) != weather_description(0)
    assert isinstance(weather_description(999), str)


def test_weather_rejects_invalid_coordinates():
    with pytest.raises(ValueError):
        _finite(float("nan"), "latitude", -90, 90)
    with pytest.raises(ValueError):
        _finite(999, "longitude", -180, 180)
