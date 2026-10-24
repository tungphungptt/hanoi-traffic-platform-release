from app.traffic_baseline import forecast_from_history


def build_history():
    start = 1_000_000.0
    return [
        {"observed_epoch": start + i * 120, "ci": 0.20 + i * 0.025, "speed_kmh": 35 - i}
        for i in range(5)
    ]


def test_baseline_needs_multiple_real_observations_over_time():
    result = forecast_from_history([], now_epoch=1_001_000)
    assert result["available"] is False
    assert result["reason"] == "insufficient_samples"
    assert result["predictions"] == []


def test_baseline_returns_five_and_ten_minute_reference_only_with_enough_history():
    history = build_history()
    result = forecast_from_history(history, now_epoch=history[-1]["observed_epoch"] + 10)
    assert result["available"] is True
    assert result["validated"] is False
    assert [p["horizon_minutes"] for p in result["predictions"]] == [5, 10]
    assert result["predictions"][1]["ci"] >= result["predictions"][0]["ci"]


def test_baseline_refuses_stale_telemetry():
    history = build_history()
    result = forecast_from_history(history, now_epoch=history[-1]["observed_epoch"] + 400)
    assert result["available"] is False
    assert result["reason"] == "stale_telemetry"
    assert result["predictions"] == []
