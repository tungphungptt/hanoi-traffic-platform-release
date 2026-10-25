"""Regression coverage for the shared three-level continuous CI scale."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.traffic_baseline import forecast_from_history
from traffic_brain.providers import _label_for_ci
from traffic_brain.traffic_levels import DEFAULT_THRESHOLDS, STATE_LABELS, level_for_ci
from traffic_brain.video_review import TemporalStateStabilizer, classify_visual_state, project_trend_baseline


CASES = [
    (0.00, "free"), (0.29, "free"), (0.299999, "free"),
    (0.30, "busy"), (0.54, "busy"), (0.549999, "busy"),
    (0.55, "gridlock"), (0.65, "gridlock"), (0.77, "gridlock"), (0.78, "gridlock"), (1.00, "gridlock"),
]


@pytest.mark.parametrize("ci,key", CASES)
def test_shared_provider_and_video_initial_state_use_identical_boundaries(ci, key):
    label = STATE_LABELS[key]
    assert level_for_ci(ci) == (key, label)
    assert _label_for_ci(ci, DEFAULT_THRESHOLDS) == (label, key)
    assert TemporalStateStabilizer().update(ci, 0) == (ci, key, label)


@pytest.mark.parametrize("ci,key", CASES)
def test_local_onnx_prediction_and_api_keep_scores_with_three_labels(monkeypatch, ci, key):
    # Isolate the presentation layer from learned weights; do not retrain or edit models.
    runtime = main.ArtifactRuntime.__new__(main.ArtifactRuntime)
    runtime.manifest = copy.deepcopy(main.DEFAULT_MANIFEST)
    runtime.forecaster_session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="features", shape=[1, 8])],
        run=lambda *_: [np.array([[ci, ci, ci]])],
    )
    monkeypatch.setattr(main.forecast_providers, "runtime", runtime)
    monkeypatch.setenv("FORECAST_PROVIDER", "local_onnx")
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    response = TestClient(main.app).post("/api/forecast", json={"features": {}})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert all(p["ci"] == ci and p["level_key"] == key and p["level"] == STATE_LABELS[key] for p in predictions)


@pytest.mark.parametrize("ci,key", [(0.29, "free"), (0.30, "busy"), (0.54, "busy"), (0.55, "gridlock"), (1.00, "gridlock")])
def test_telemetry_and_clip_trends_use_same_scale(ci, key):
    history = [{"observed_epoch": 1000 + t, "ci": ci} for t in range(0, 481, 120)]
    result = forecast_from_history(history, now_epoch=1481)
    assert result["available"]
    assert result["current_level_key"] == key
    assert all(p["level_key"] == key and p["level"] == STATE_LABELS[key] for p in result["predictions"])
    clip = [{"timestamp_seconds": t, "ci": ci} for t in range(60)]
    result = project_trend_baseline(clip, source_duration_seconds=60, preserve_real_timing=True)
    assert result["available"]
    assert all(p["level_key"] == key and p["level"] == STATE_LABELS[key] for p in result["predictions"])


def test_visual_state_keeps_existing_score_formula_but_merges_severe_states():
    for count, occupancy, speed in [(0, 0, .05), (10, .10, .02), (16, .13, .02), (20, .22, .0001)]:
        ci = .50 * min(occupancy / .22, 1) + .32 * min(count / 20, 1) + .18 * (1 - min(speed / .045, 1))
        state = classify_visual_state(count, occupancy, speed)
        key, label = level_for_ci(ci)
        assert state["ci"] == state["raw_ci"] == ci
        assert state["state_key"] == key
        assert state["state_label"] == label


def test_dashboard_counts_severe_points_starting_at_point_55(monkeypatch):
    telemetry = {str(i): {"ci": ci} for i, ci in enumerate([.29, .30, .54, .55, .77, .78, 1])}
    monkeypatch.setattr(main, "_recent_live_telemetry", lambda: telemetry)
    monkeypatch.setattr(main, "_public_camera_rows", lambda: [])
    response = TestClient(main.app).get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["summary"]["critical_points"] == 4


def test_three_state_hysteresis_retains_causal_stabilization():
    stabilizer = TemporalStateStabilizer(hysteresis_margin=.04, min_state_duration_seconds=1, ema_alpha=1)
    for ci, timestamp, expected in [(.4, 0, "busy"), (.57, 1, "busy"), (.60, 2, "busy"),
                                    (.60, 3.1, "gridlock"), (.53, 4, "gridlock"),
                                    (.50, 5, "gridlock"), (.50, 6.1, "busy")]:
        smoothed_ci, key, label = stabilizer.update(ci, timestamp)
        assert stabilizer.current_state_key == expected
        assert (key, label) == level_for_ci(smoothed_ci)


def test_manifest_compatibility_alias_does_not_reintroduce_fourth_state():
    manifest = main._normalize_manifest_payload(copy.deepcopy(main.DEFAULT_MANIFEST))
    assert manifest["forecaster_config"]["thresholds"] == {"busy": .30, "heavy": .55, "gridlock": .55}
    assert set(STATE_LABELS) == {"free", "busy", "gridlock"}
    manifest["forecaster_config"]["thresholds"]["gridlock"] = .78
    with pytest.raises(ValueError, match="3 mức"):
        main._normalize_manifest_payload(manifest)
