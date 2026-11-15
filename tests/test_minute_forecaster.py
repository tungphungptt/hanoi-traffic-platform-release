import math
import random

import numpy as np

from traffic_brain import minute_forecaster as mf


def episode_timeline(seed: int, duration: int = 40 * 60):
    """1 Hz rows with congestion episodes that build up over several minutes."""
    rng = random.Random(seed)
    level = [0.2] * duration
    start = rng.randint(200, 400)
    while start < duration:
        rise, hold, fall = rng.randint(360, 600), rng.randint(180, 420), rng.randint(120, 240)
        peak = rng.uniform(0.62, 0.85)
        for i in range(rise):
            if start + i < duration:
                level[start + i] = 0.2 + (peak - 0.2) * (i / rise) ** 1.5
        for i in range(hold):
            if start + rise + i < duration:
                level[start + rise + i] = peak
        for i in range(fall):
            if start + rise + hold + i < duration:
                level[start + rise + hold + i] = peak - (peak - 0.2) * i / fall
        start += rise + hold + fall + rng.randint(240, 600)
    rows = []
    for t in range(duration):
        ci = min(1.0, max(0.0, level[t] + rng.gauss(0, 0.07)))
        rows.append({"timestamp_seconds": t, "raw_ci": ci, "occupancy_ratio": ci * 0.2,
                     "vehicle_count": int(ci * 20), "normalized_speed": max(0.0, 0.04 * (1 - ci))})
    return rows


def predictions_by_anchor(result):
    return {a["t"]: [(h["horizon_minutes"], h["ci"], h["p_gridlock"]) for h in a["horizons"]] for a in result["anchors"]}


def test_short_video_is_refused_with_required_duration():
    rows = episode_timeline(1, duration=6 * 60)
    result = mf.build_minute_forecast(rows, duration_seconds=360, timing_confirmed=True)
    assert result["available"] is False
    assert result["reason"] == "video_too_short"
    assert "phút" in result["message"]


def test_forecasts_never_use_data_after_the_anchor_time():
    rows = episode_timeline(2)
    cutoff = 20 * 60
    original = mf.build_minute_forecast(rows, duration_seconds=len(rows), timing_confirmed=True)
    tampered_rows = [dict(row) for row in rows]
    for row in tampered_rows:
        if row["timestamp_seconds"] > cutoff:
            row["raw_ci"] = 1.0 - row["raw_ci"]
            row["vehicle_count"] = 0
    tampered = mf.build_minute_forecast(tampered_rows, duration_seconds=len(rows), timing_confirmed=True)
    before = {t: v for t, v in predictions_by_anchor(original).items() if t <= cutoff}
    after = {t: v for t, v in predictions_by_anchor(tampered).items() if t <= cutoff}
    assert before and before == after


def test_horizon_items_are_consistent():
    result = mf.build_minute_forecast(episode_timeline(3), duration_seconds=40 * 60, timing_confirmed=False)
    assert result["available"] is True
    assert "thời gian thực" in result["message"]
    for anchor in result["anchors"]:
        assert [h["horizon_minutes"] for h in anchor["horizons"]] == [5, 10, 15]
        for item in anchor["horizons"]:
            assert math.isclose(item["target_t"], anchor["t"] + item["horizon_minutes"] * 60)
            assert 0 <= item["lower"] <= item["ci"] <= item["upper"] <= 1
            assert item["warning"] == (item["p_gridlock"] >= mf.WARNING_PROBABILITY)
            if item["target_t"] > 40 * 60 - 1:
                assert item["actual_ci"] is None


def test_constant_traffic_is_forecast_as_constant():
    rows = [{"timestamp_seconds": t, "raw_ci": 0.4, "occupancy_ratio": 0.08, "vehicle_count": 8,
             "normalized_speed": 0.02} for t in range(20 * 60)]
    result = mf.build_minute_forecast(rows, duration_seconds=1200, timing_confirmed=True)
    metric = result["metrics"][0]
    assert metric["evaluated"] > 0
    assert metric["mae_model"] < 1e-6
    assert metric["mae_persistence"] < 1e-6


def test_library_of_other_videos_improves_on_persistence_for_build_up_episodes():
    library = [mf.build_observed_series(episode_timeline(100 + i)) for i in range(3)]
    skills = []
    for seed in range(3):
        rows = episode_timeline(seed)
        result = mf.build_minute_forecast(rows, duration_seconds=len(rows), timing_confirmed=True,
                                          library=library, library_videos=len(library))
        assert result["method"]["library_videos"] == 3
        skills.append(result["metrics"][0]["skill_vs_persistence"])
    assert np.mean(skills) > 0.1


def test_library_record_round_trip_and_rejects_other_grids():
    series = mf.build_observed_series(episode_timeline(4, duration=12 * 60))
    record = mf.series_record(series, video_key="abc", filename="clip.mp4")
    parsed = mf.record_to_series(record)
    assert parsed is not None
    assert np.allclose(np.nan_to_num(parsed["ci"]), np.nan_to_num(series["ci"]), atol=1e-6)
    assert mf.record_to_series({**record, "step_seconds": 30}) is None
    assert mf.record_to_series({**record, "contract": "other"}) is None
    assert mf.record_to_series({**record, "ci": record["ci"][:-1]}) is None


def test_highlights_are_real_outcomes():
    library = [mf.build_observed_series(episode_timeline(100 + i)) for i in range(3)]
    result = mf.build_minute_forecast(episode_timeline(0), duration_seconds=40 * 60, timing_confirmed=True,
                                      library=library, library_videos=3)
    for hit in result["highlights"]["hits"]:
        assert hit["observed_ci"] < mf.SEVERE_CI <= hit["actual_ci"]
        assert hit["p_gridlock"] >= mf.WARNING_PROBABILITY
    for miss in result["highlights"]["misses"]:
        assert miss["actual_ci"] >= mf.SEVERE_CI > 0 and miss["p_gridlock"] < mf.WARNING_PROBABILITY


def test_bundled_library_beats_persistence_leave_one_video_out():
    """Regression guard on the shipped training data (real street videos)."""
    import json
    from pathlib import Path
    files = sorted((Path(__file__).resolve().parents[1] / "models" / "forecast_library").glob("*.json"))
    series = [mf.record_to_series(json.loads(p.read_text(encoding="utf-8"))) for p in files]
    assert len(series) >= 3
    totals = {h: [0.0, 0.0] for h in mf.HORIZONS_MINUTES}
    for i, target in enumerate(series):
        anchors = mf.forecast_series(target, library=[s for j, s in enumerate(series) if j != i])
        for metric in mf.evaluate_anchors(anchors):
            if metric.get("evaluated"):
                totals[metric["horizon_minutes"]][0] += metric["mae_model"] * metric["evaluated"]
                totals[metric["horizon_minutes"]][1] += metric["mae_persistence"] * metric["evaluated"]
    for h, (model, persistence) in totals.items():
        assert model < persistence, f"+{h} min forecast is not better than keeping the current level"
