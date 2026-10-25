from __future__ import annotations

import pytest

from traffic_brain.analytics import (
    AnalyticsInputError,
    TrafficAnalyticsEngine,
    adaptive_frame_stride,
    segments_intersect,
)


def test_adaptive_stride_caps_detector_rate_for_cpu():
    result = adaptive_frame_stride(30, 6)
    assert result == {"frame_stride": 5, "sample_fps": 6.0, "source_fps": 30.0}
    assert adaptive_frame_stride(4, 6)["frame_stride"] == 1


def test_adaptive_stride_rejects_nan():
    with pytest.raises(AnalyticsInputError):
        adaptive_frame_stride(float("nan"), 6)


def test_segments_intersection_handles_crossing_and_non_crossing():
    assert segments_intersect((0.4, 0.2), (0.6, 0.8), (0.5, 0), (0.5, 1))
    assert not segments_intersect((0.1, 0.1), (0.2, 0.2), (0.5, 0), (0.5, 1))


def test_counting_line_is_motion_gated_and_deduplicated_per_track():
    engine = TrafficAnalyticsEngine()
    line = [{"line_id": "northbound", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}]

    engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .40, "y": .50}], timestamp_epoch=100, counting_lines=line)
    before = engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .45, "y": .50}], timestamp_epoch=101, counting_lines=line)
    assert before["new_crossings"] == 0

    crossing = engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .60, "y": .50}], timestamp_epoch=102, counting_lines=line)
    assert crossing["new_crossings"] == 1
    assert crossing["counting_lines"]["northbound"] == 1

    again = engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .70, "y": .50}], timestamp_epoch=103, counting_lines=line)
    assert again["new_crossings"] == 0
    assert again["counting_lines"]["northbound"] == 1


def test_engine_rejects_duplicate_track_ids_in_same_frame():
    engine = TrafficAnalyticsEngine()
    tracks = [
        {"track_id": "same", "class_name": "car", "x": .1, "y": .1},
        {"track_id": "same", "class_name": "car", "x": .2, "y": .2},
    ]
    with pytest.raises(AnalyticsInputError, match="bị lặp"):
        engine.process_frame(tracks, timestamp_epoch=1)


def test_confidence_and_event_output_are_bounded_and_operator_reviewed():
    engine = TrafficAnalyticsEngine(history_size=10, event_size=5)
    payload = engine.process_frame(
        [{"track_id": "c", "class_name": "car", "x": .1, "y": .1, "confidence": .9}],
        timestamp_epoch=10,
    )
    assert payload["active_tracks"] == 1
    assert payload["counts_by_class"] == {"car": 1}
    assert payload["quality"]["operator_review_required_for_alerts"] is True
    assert payload["quality"]["coordinate_system"] == "normalized_0_1"


def test_api_tracker_ingestion_contract_and_events(monkeypatch):
    import app.main as main
    from fastapi.testclient import TestClient

    main.ANALYTICS_ENGINES.clear()
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    client = TestClient(main.app)
    payload = {
        "camera_id": "HN-001",
        "timestamp": "2026-10-23T03:00:00Z",
        "tracks": [{"track_id": "vehicle-1", "class_name": "car", "x": .4, "y": .4, "confidence": .91}],
        "counting_lines": [{"line_id": "gate-1", "x1": .5, "y1": 0, "x2": .5, "y2": 1}],
    }
    response = client.post("/api/analytics/ingest-frame", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["camera_id"] == "HN-001"
    assert body["analytics"]["active_tracks"] == 1
    assert "không tự chạy detector/RTSP" in body["notice"]
    events = client.get("/api/analytics/cameras/HN-001/events")
    assert events.status_code == 200
    assert events.json()["persistent"] is False


def test_api_tracker_ingestion_rejects_unknown_camera():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post("/api/analytics/ingest-frame", json={"camera_id": "NOT-A-CAMERA", "tracks": []})
    assert response.status_code == 404


def test_api_tracker_ingestion_rejects_out_of_range_coordinates():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post("/api/analytics/ingest-frame", json={
        "camera_id": "HN-001", "tracks": [{"track_id": "x", "class_name": "car", "x": 1.2, "y": .3}]
    })
    assert response.status_code == 422


def test_long_observation_gap_does_not_fabricate_a_line_crossing():
    engine = TrafficAnalyticsEngine()
    line = [{"line_id": "gate", "x1": .5, "y1": 0, "x2": .5, "y2": 1}]
    engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .4, "y": .5}], timestamp_epoch=10, counting_lines=line)
    result = engine.process_frame([{"track_id": "car-1", "class_name": "car", "x": .6, "y": .5}], timestamp_epoch=20, counting_lines=line)
    assert result["new_crossings"] == 0


def test_counted_track_keys_are_bounded():
    engine = TrafficAnalyticsEngine(max_counted_keys=2)
    line = [{"line_id": "gate", "x1": .5, "y1": 0, "x2": .5, "y2": 1}]
    for index, track_id in enumerate(("a", "b", "c")):
        t = 100 + index * 10
        engine.process_frame([{"track_id": track_id, "class_name": "car", "x": .4, "y": .5}], timestamp_epoch=t, counting_lines=line)
        engine.process_frame([{"track_id": track_id, "class_name": "car", "x": .6, "y": .5}], timestamp_epoch=t + 1, counting_lines=line)
    assert len(engine._counted) <= 2
    assert engine._counts_by_line["gate"] == 3


def test_public_demo_blocks_tracker_ingest(monkeypatch):
    import app.main as main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", True)
    client = TestClient(main.app)
    response = client.post("/api/analytics/ingest-frame", json={"camera_id": "HN-001", "tracks": []})
    assert response.status_code == 403
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)


def test_engine_rejects_duplicate_or_out_of_order_frame_timestamps():
    engine = TrafficAnalyticsEngine()
    engine.process_frame([], timestamp_epoch=50)
    with pytest.raises(AnalyticsInputError, match="tăng nghiêm ngặt"):
        engine.process_frame([], timestamp_epoch=50)
    with pytest.raises(AnalyticsInputError, match="tăng nghiêm ngặt"):
        engine.process_frame([], timestamp_epoch=49)
