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


def test_baseline_refuses_a_large_gap_even_when_latest_telemetry_is_fresh():
    start = 1_000_000.0
    history = [
        {"observed_epoch": start + offset, "ci": 0.2 + index * 0.025}
        for index, offset in enumerate((0, 60, 120, 420, 480))
    ]

    result = forecast_from_history(history, now_epoch=history[-1]["observed_epoch"] + 10)

    assert result["available"] is False
    assert result["reason"] == "gapped_history"
    assert result["predictions"] == []


def test_baseline_accepts_history_with_gaps_at_the_freshness_limit():
    start = 1_000_000.0
    history = [
        {"observed_epoch": start + index * 180, "ci": 0.2 + index * 0.025}
        for index in range(5)
    ]

    result = forecast_from_history(history, now_epoch=history[-1]["observed_epoch"] + 10)

    assert result["available"] is True
    assert result["sample_count"] == 5
