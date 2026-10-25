"""Audit regressions across HTTP serialization, frontend rendering and video output."""
import copy
import csv
import io
import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
import traffic_brain.video_review as review
from app.traffic_baseline import forecast_from_history
from traffic_brain.traffic_levels import STATE_LABELS, level_for_ci


BOUNDARIES = [(0.299999, "free"), (0.30, "busy"), (0.549999, "busy"), (0.55, "gridlock")]


@pytest.mark.parametrize("provider", ["local_onnx", "remote_http"])
def test_http_predictions_roundtrip_to_real_frontend_without_changing_ci_or_label(monkeypatch, provider):
    monkeypatch.setenv("FORECAST_PROVIDER", provider)
    monkeypatch.setenv("FORECAST_API_URL", "https://forecast.example.test/predict")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    features = {key: 0 for key in main.DEFAULT_MANIFEST["feature_schema"]["ordered_input_keys"]}
    responses = []
    for ci, key in BOUNDARIES:
        if provider == "local_onnx":
            runtime = main.ArtifactRuntime.__new__(main.ArtifactRuntime)
            runtime.manifest = copy.deepcopy(main.DEFAULT_MANIFEST)
            runtime.forecaster_session = SimpleNamespace(
                get_inputs=lambda: [SimpleNamespace(name="features", shape=[1, len(features)])],
                run=lambda *_: [np.array([[ci, ci, ci]])],
            )
            monkeypatch.setattr(main.forecast_providers, "runtime", runtime)
        else:
            def urlopen(*_args, **_kwargs):
                body = io.BytesIO(json.dumps({"predictions": [
                    {"horizon_minutes": h, "ci": ci} for h in (5, 10, 15)
                ]}).encode())
                body.status = 200
                return body
            monkeypatch.setattr("traffic_brain.providers.urlopen", urlopen)
        response = TestClient(main.app).post("/api/forecast", json={"features": features})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["predictions"]) == 3
        assert all(p["ci"] == ci and (p["level_key"], p["level"]) == (key, STATE_LABELS[key])
                   for p in payload["predictions"])
        responses.append(payload)

    # Run the actual frontend helpers and renderForecast on serialized HTTP responses.
    script = """
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('static/app.js','utf8').split('\\nif (!window.__weatherTrafficPollerV1)')[0];
const nodes=new Map();
const el=id=>{if(!nodes.has(id))nodes.set(id,{style:{},setAttribute(){}});return nodes.get(id);};
const ctx=vm.createContext({console,document:{querySelector:el},window:{},AbortController,FormData,Blob,setTimeout,clearTimeout});
vm.runInContext(source,ctx);
for(const forecast of JSON.parse(fs.readFileSync(0,'utf8'))){
  ctx.forecast=forecast;
  vm.runInContext('renderForecast(forecast,null)',ctx);
  const p=forecast.predictions[0];
  assert.equal(vm.runInContext('severityFromCI(forecast.predictions[0].ci)',ctx),p.level_key);
  assert.equal(vm.runInContext('levelVN(severityFromCI(forecast.predictions[0].ci))',ctx),p.level);
  assert.ok(el('#horizonList').innerHTML.includes('class="'+p.level_key+'"'));
}
"""
    subprocess.run(["node", "-e", script], input=json.dumps(responses), text=True,
                   capture_output=True, check=True, timeout=15)


@pytest.mark.parametrize("ci,key", BOUNDARIES)
def test_dashboard_average_does_not_round_across_a_classification_boundary(monkeypatch, ci, key):
    monkeypatch.setattr(main, "_recent_live_telemetry", lambda: {"HN-001": {"ci": ci}})
    monkeypatch.setattr(main, "_public_camera_rows", lambda: [])
    response = TestClient(main.app).get("/api/dashboard")
    assert response.status_code == 200
    average = response.json()["summary"]["average_ci"]
    assert average == ci
    assert level_for_ci(average)[0] == key


@pytest.mark.parametrize("ci,key", BOUNDARIES)
def test_baseline_and_csv_preserve_ci_next_to_boundaries(ci, key, tmp_path):
    result = forecast_from_history([{"observed_epoch": 1000 + t, "ci": ci} for t in range(0, 481, 120)], now_epoch=1481)
    assert result["current_ci"] == ci
    clip = review.project_trend_baseline([{"timestamp_seconds": t, "ci": ci} for t in range(60)],
                                        source_duration_seconds=60, preserve_real_timing=True)
    for prediction in result["predictions"] + clip["predictions"]:
        assert prediction["ci"] == pytest.approx(ci, abs=1e-12)
        assert (prediction["level_key"], prediction["level"]) == level_for_ci(prediction["ci"])
    path = tmp_path / "evidence.csv"
    review.export_evidence_csv(path, [{"ci": ci, "raw_ci": ci, "state_key": key, "state_label": STATE_LABELS[key]}])
    row = next(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    assert float(row["ci"]) == ci
    assert level_for_ci(float(row["ci"])) == (row["state_key"], row["state_label"])


@pytest.mark.parametrize("start,steady,expected", [(.54, .56, "gridlock"), (.60, .54, "busy")])
def test_official_video_label_follows_ci_after_1000_steady_samples(start, steady, expected):
    stabilizer = review.TemporalStateStabilizer()
    stabilizer.update(start, 0)
    for timestamp in range(1, 1001):
        ci, key, label = stabilizer.update(steady, timestamp)
        assert (key, label) == level_for_ci(ci)
    assert key == expected
    # The hysteresis state can remain stable, but must not overwrite the official result.
    assert stabilizer.current_state_key != key


@pytest.mark.parametrize("boundary", [.30, .55])
def test_official_video_labels_follow_repeated_crossings_without_hysteresis_override(boundary):
    stabilizer = review.TemporalStateStabilizer(ema_alpha=1)
    stream = [boundary - .01, boundary, boundary + .01, boundary - .01] * 10
    for timestamp, raw_ci in enumerate(stream):
        ci, key, label = stabilizer.update(raw_ci, timestamp)
        assert ci == raw_ci
        assert (key, label) == level_for_ci(ci)


def test_video_pipeline_uses_official_labels_in_timeline_summary_frames_and_csv(monkeypatch, tmp_path):
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 1, (96, 64))
    assert writer.isOpened(), "MJPG video writer is required for this regression"
    stream = [.54, .56, .56, .54, .299999, .30, .549999, .55]
    for _ in stream:
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(review, "preprocess_frame", lambda frame: frame)
    monkeypatch.setattr(review, "_detect", lambda *_: [])
    monkeypatch.setattr(review, "_update_tracks", lambda *_: ({}, 1, []))
    states = iter(stream)
    classify = review.classify_visual_state
    def controlled_state(*_args):
        ci = next(states)
        # Produce the chosen CI through the original, unchanged occupancy/count formula.
        return classify(0, ci * .22 / .62)
    monkeypatch.setattr(review, "classify_visual_state", controlled_state)
    bundle = {"manifest": {"sequence_length": 100}, "class_ids": {2}, "detector": object(), "detector_input_name": "images"}
    result = review.analyze_video(source, bundle, tmp_path / "output", stabilizer=review.TemporalStateStabilizer(ema_alpha=1))
    assert len(result["timeline"]) == len(stream)
    for row in result["timeline"] + result["frames"]:
        assert (row["state_key"], row["state_label"]) == level_for_ci(row["ci"])
        assert (row["raw_state_key"], row["raw_state_label"]) == level_for_ci(row["raw_ci"])
    assert any(row["smoothed_state_key"] != row["state_key"] for row in result["timeline"])
    assert result["summary"]["current_state"] == level_for_ci(result["summary"]["current_ci"])[1]
    for name in ("evidence.csv", "timeline.csv"):
        for row in csv.DictReader((tmp_path / "output" / name).read_text(encoding="utf-8-sig").splitlines()):
            assert (row["state_key"], row["state_label"]) == level_for_ci(float(row["ci"]))


@pytest.mark.parametrize("raw_ci", ["NaN", "Infinity", "-Infinity", "-0.01", "1.01", "null", "1e309", "-1e309"])
def test_telemetry_invalid_ci_returns_strict_json_422_over_http(raw_ci):
    response = TestClient(main.app, raise_server_exceptions=False).post(
        "/api/traffic/telemetry", content='{"camera_id":"HN-001","ci":' + raw_ci + '}',
        headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/json")
    payload = json.loads(response.text, parse_constant=lambda value: pytest.fail("Non-JSON constant: " + value))
    assert any(error["loc"] == ["body", "ci"] for error in payload["detail"])


@pytest.mark.parametrize("ci", [0, .29, .30, .54, .55, .65, 1])
def test_valid_telemetry_ci_keeps_its_value_and_existing_response(monkeypatch, ci):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "CAMERAS", [{"id": "HN-001", "location_verified": True, "lat": 21.002, "lng": 105.82}])
    monkeypatch.setattr(main, "LIVE_TELEMETRY", {})
    monkeypatch.setattr(main, "LIVE_TELEMETRY_HISTORY", {})
    response = TestClient(main.app).post("/api/traffic/telemetry", json={"camera_id": "HN-001", "ci": ci})
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert response.json()["stored"] is True
    assert main.LIVE_TELEMETRY["HN-001"]["ci"] == ci
