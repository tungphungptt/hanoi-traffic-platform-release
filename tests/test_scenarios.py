from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from traffic_brain.routing import score_routes
from traffic_brain.video_review import (
    VideoReviewError,
    analyze_video,
    classify_visual_state,
    project_trend_baseline,
)

client = TestClient(main.app)


def test_visual_state_logic_returns_explainable_levels():
    free = classify_visual_state(0, 0.01, 0.08)
    busy = classify_visual_state(10, 0.10, 0.02)
    heavy = classify_visual_state(16, 0.13, 0.02)
    gridlock = classify_visual_state(20, 0.22, 0.0001)
    assert free["state_key"] == "free"
    assert busy["state_key"] == "busy"
    assert heavy["state_key"] == "gridlock"
    assert gridlock["state_key"] == "gridlock"
    assert free["method"] == "explainable_visual_rule_baseline"
    assert all(0 <= x["ci"] <= 1 for x in (free, busy, heavy, gridlock))


@pytest.mark.parametrize("kwargs", [
    {"vehicle_count": -1, "occupancy_ratio": 0.1},
    {"vehicle_count": 3, "occupancy_ratio": 1.1},
    {"vehicle_count": 3, "occupancy_ratio": float("nan")},
    {"vehicle_count": 3, "occupancy_ratio": 0.2, "normalized_speed": -1},
])
def test_visual_state_logic_rejects_invalid_inputs(kwargs):
    with pytest.raises(VideoReviewError):
        classify_visual_state(**kwargs)


def test_trend_baseline_is_gated_by_real_timing_confirmation():
    timeline = [{"timestamp_seconds": i, "ci": 0.2 + i * 0.001} for i in range(120)]
    result = project_trend_baseline(timeline, source_duration_seconds=120, preserve_real_timing=False)
    assert result["available"] is False
    assert result["predictions"] == []
    assert "không ngoại suy" in result["message"].lower()


def test_trend_baseline_requires_enough_history_and_returns_only_5_10_minutes():
    too_short = project_trend_baseline(
        [{"timestamp_seconds": i, "ci": 0.4} for i in range(19)],
        source_duration_seconds=60,
        preserve_real_timing=True,
    )
    assert not too_short["available"]
    timeline = [{"timestamp_seconds": i, "ci": min(0.9, 0.2 + i * 0.002)} for i in range(120)]
    result = project_trend_baseline(timeline, source_duration_seconds=120, preserve_real_timing=True)
    assert result["available"]
    assert result["mode"] == "explainable_trend_baseline"
    assert [row["horizon_minutes"] for row in result["predictions"]] == [5, 10]
    assert "không phải mô hình" in result["message"].lower()


def test_analyze_video_creates_real_sampled_frame_and_explicit_limits(monkeypatch, tmp_path):
    source = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG writer")
    for idx in range(30):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 1] = idx * 3
        writer.write(frame)
    writer.release()

    monkeypatch.setattr("traffic_brain.video_review.preprocess_frame", lambda frame: frame)
    monkeypatch.setattr("traffic_brain.video_review._detect", lambda *args, **kwargs: [])
    monkeypatch.setattr("traffic_brain.video_review._update_tracks", lambda detections, tracks, next_id, sample_index: ({}, 1, []))
    bundle = {"manifest": {"sequence_length": 100}, "class_ids": {2}, "detector": object(),
              "detector_input_name": "images"}
    result = analyze_video(source, bundle, tmp_path / "result", preserve_real_timing=False, sample_fps=1.0)
    assert result["status"] == "succeeded"
    assert result["summary"]["sampled_frames"] >= 1
    assert result["frames"]
    assert result["hanoi_forecast"]["available"] is False
    assert result["hanoi_forecast"]["mode"] == "insufficient_history"
    assert (tmp_path / "result" / "frames" / result["frames"][0]["filename"]).is_file()
    assert any("pixel/giây" in note for note in result["limitations"])
    saved = json.loads((tmp_path / "result" / "result.json").read_text(encoding="utf-8"))
    assert saved["pipeline"] == result["pipeline"]


def test_video_from_data_rejects_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "AICC_VIDEO_ENABLED", True)
    response = client.post("/api/scenarios/video-review-from-data", json={
        "path": "../README.md", "camera_id": "HN-001", "preserve_real_timing": False
    })
    assert response.status_code == 400
    assert "đường dẫn" in response.json()["detail"].lower()


def test_public_demo_blocks_video_review(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", True)
    monkeypatch.setattr(main, "AICC_VIDEO_ENABLED", True)
    monkeypatch.setattr(main, "PUBLIC_VIDEO_REVIEW_ENABLED", False)
    response = client.post("/api/scenarios/video-review-from-data", json={"path": "input.mp4"})
    assert response.status_code == 403
    protected_response = client.post("/api/aicc/predict-video", files={"file": ("clip.mp4", b"video")})
    assert protected_response.status_code == 403
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)


def test_public_demo_allows_only_capped_uploaded_video_review(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", True)
    monkeypatch.setattr(main, "AICC_VIDEO_ENABLED", False)
    monkeypatch.setattr(main, "PUBLIC_VIDEO_REVIEW_ENABLED", True)
    status = client.get("/api/aicc/status").json()
    assert status["upload_limit_mb"] == 500
    assert status["upload_limit_mib"] == 500
    assert status["upload_limit_bytes"] == 524_288_000
    assert status["upload_max_duration_seconds"] == 360

    source = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG writer")
    for _ in range(5):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()

    def enqueue(video_path, filename, camera_id, preserve_real_timing, cleanup):
        assert video_path.is_file()
        assert filename == "short.avi"
        assert camera_id is None
        assert preserve_real_timing is False
        assert cleanup is True
        video_path.unlink()
        return {"job_id": "a" * 32, "status": "queued"}

    monkeypatch.setattr(main, "_load_video_review_models", lambda: None)
    monkeypatch.setattr(main, "_enqueue_video_review_job", enqueue)
    response = client.post(
        "/api/scenarios/video-review",
        files={"file": ("short.avi", source.read_bytes(), "video/x-msvideo")},
    )
    assert response.status_code == 202, response.text
    assert response.json()["job_id"] == "a" * 32

    monkeypatch.setattr(main, "MAX_PUBLIC_VIDEO_REVIEW_BYTES", 1)
    oversized = client.post(
        "/api/scenarios/video-review",
        files={"file": ("short.avi", source.read_bytes(), "video/x-msvideo")},
    )
    assert oversized.status_code == 413
    monkeypatch.setattr(main, "MAX_PUBLIC_VIDEO_REVIEW_BYTES", 500 * 1024 * 1024)
    monkeypatch.setattr(main, "MAX_PUBLIC_VIDEO_REVIEW_DURATION_SECONDS", 360)

    class FakeCapture:
        def __init__(self, frame_count):
            self.frame_count = frame_count

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FPS:
                return 30
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return self.frame_count
            return 0

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture(30 * 360))
    six_minutes = client.post(
        "/api/scenarios/video-review",
        files={"file": ("short.avi", source.read_bytes(), "video/x-msvideo")},
    )
    assert six_minutes.status_code == 202, six_minutes.text

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: FakeCapture(30 * 361))
    over_six_minutes = client.post(
        "/api/scenarios/video-review",
        files={"file": ("short.avi", source.read_bytes(), "video/x-msvideo")},
    )
    assert over_six_minutes.status_code == 413


def test_frame_endpoint_rejects_traversal_and_non_jpeg(monkeypatch, tmp_path):
    response = client.get(f"/api/scenarios/video-reviews/{'a' * 32}/frames/..%2F..%2FREADME.md")
    assert response.status_code == 404
    response = client.get(f"/api/scenarios/video-reviews/{'a' * 32}/frames/not-a-frame.png")
    assert response.status_code == 404


def test_route_scoring_falls_back_to_static_duration_without_live_telemetry():
    result = score_routes([
        {"distance": 1000, "duration": 500, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.00], [105.83, 21.01]]}},
        {"distance": 1400, "duration": 420, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.00], [105.835, 21.012]]}},
    ], [], now_epoch=1000)
    assert result["traffic_aware"] is False
    assert result["recommendation_basis"] == "osrm_reference_duration_only"
    assert result["recommended_route_id"] == "R2"
    assert "chưa thể khẳng định ít tắc hơn" in result["message"].lower()


def test_route_api_uses_road_network_geometry_and_disclaims_non_live_routing(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "fetch_route_alternatives", lambda *args, **kwargs: [
        {"distance": 2300, "duration": 400, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.002], [105.825, 21.01]]}},
        {"distance": 2700, "duration": 360, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.002], [105.83, 21.012]]}},
    ])
    response = client.post("/api/routes/alternatives", json={
        "origin": {"label": "A", "camera_id": "HN-001", "lat": 21.002, "lng": 105.82},
        "destination": {"label": "B", "camera_id": "HN-002", "lat": 21.036, "lng": 105.79},
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["route_provider"] == "OSRM road network"
    assert payload["traffic_aware"] is False
    assert payload["routing_mode"] == "base_network_duration_only"
    assert len(payload["routes"]) == 2
    assert "chưa thể khẳng định ít tắc hơn" in payload["message"].lower()


def test_route_api_rejects_coordinates_outside_hanoi(monkeypatch):
    response = client.post("/api/routes/alternatives", json={
        "origin": {"label": "Outside", "lat": 13.75, "lng": 100.5},
        "destination": {"label": "Hanoi", "lat": 21.0, "lng": 105.8},
    })
    assert response.status_code == 422


def test_route_api_attaches_video_analysis_only_when_origin_camera_matches(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "fetch_route_alternatives", lambda *args, **kwargs: [
        {"distance": 1200, "duration": 300, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.002], [105.83, 21.01]]}},
        {"distance": 1300, "duration": 280, "geometry": {"type": "LineString", "coordinates": [[105.82, 21.002], [105.831, 21.01]]}},
    ])
    job_id = "f" * 32
    main.VIDEO_REVIEW_JOBS[job_id] = {"status": "succeeded", "result": {
        "source_camera_id": "HN-001", "summary": {"current_state": "Đông đúc", "current_ci": 0.4},
        "trend_forecast": {"available": False, "predictions": []}
    }}
    try:
        same = client.post("/api/routes/alternatives", json={
            "origin": {"label": "A", "camera_id": "HN-001", "lat": 21.002, "lng": 105.82},
            "destination": {"label": "B", "lat": 21.036, "lng": 105.79}, "analysis_job_id": job_id
        })
        assert same.status_code == 200
        assert same.json()["source_analysis"]["available"] is True
        other = client.post("/api/routes/alternatives", json={
            "origin": {"label": "Other", "camera_id": "HN-002", "lat": 21.036, "lng": 105.79},
            "destination": {"label": "B", "lat": 21.002, "lng": 105.82}, "analysis_job_id": job_id
        })
        assert other.status_code == 200
        assert other.json()["source_analysis"]["available"] is False
        assert "không dùng để suy luận" in other.json()["source_analysis"]["notice"].lower()
    finally:
        main.VIDEO_REVIEW_JOBS.pop(job_id, None)


def test_camera_telemetry_requires_verified_camera_location(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    response = client.post("/api/traffic/telemetry", json={
        "camera_id": "HN-001", "source": "rtsp", "ci": 0.7,
        "observed_at": "2026-10-23T07:00:00Z"
    })
    assert response.status_code == 409
    assert "chưa được xác minh" in response.json()["detail"].lower()


def test_v5_ui_explains_video_frames_route_candidates_and_hides_legacy_flow_by_default():
    html = client.get("/").text
    assert 'id="scenarioFrameGallery"' in html
    assert 'id="scenarioFindRoutesBtn"' in html
    assert 'id="scenarioRouteMap"' in html
    assert 'id="scenarioPreserveTiming"' in html
    assert 'id="scenarioDataVideo"' not in html
    assert 'id="aiccVideoFile"' not in html
    assert 'id="scenarioVideoFile"' in html
    assert "Dự báo & phương án" in html
