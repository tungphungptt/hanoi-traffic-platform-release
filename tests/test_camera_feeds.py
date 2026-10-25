from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


class _FakeCapture:
    def isOpened(self):
        return True

    def get(self, prop):
        return 25.0

    def read(self):
        return True, object()

    def release(self):
        pass


def test_camera_config_has_unique_ids_and_map_positions():
    payload = json.loads((main.DATA_DIR / "cameras.json").read_text(encoding="utf-8"))
    cameras = payload["cameras"]
    ids = [entry["id"] for entry in cameras]
    assert len(ids) >= 1
    assert len(ids) == len(set(ids))
    for camera in cameras:
        assert 3 <= camera["map_x"] <= 97
        assert 3 <= camera["map_y"] <= 97


def test_camera_api_exposes_feed_metadata():
    response = client.get("/api/cameras")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 1
    first = payload["items"][0]
    assert "map_x" in first and "map_y" in first
    assert "feed" in first
    info = client.get(f"/api/cameras/{first['id']}/feed-info")
    assert info.status_code == 200
    assert info.json()["camera_id"] == first["id"]


def test_camera_feed_upload_assigns_clip_and_supports_range(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "CAMERA_FEED_DIR", tmp_path)
    monkeypatch.setattr(main, "CAMERA_FEED_INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: _FakeCapture())

    clip = b"0123456789abcdef"
    uploaded = client.post(
        "/api/cameras/HN-001/feed",
        files={"file": ("test.mp4", clip, "video/mp4")},
    )
    assert uploaded.status_code == 200, uploaded.text
    payload = uploaded.json()
    assert payload["feed"]["available"] is True
    assert payload["feed"]["source_type"] == "assigned_clip"

    streamed = client.get("/api/cameras/HN-001/feed", headers={"Range": "bytes=0-3"})
    assert streamed.status_code == 206
    assert streamed.content == clip[:4]
    assert streamed.headers.get("content-range") == f"bytes 0-3/{len(clip)}"


def test_camera_feed_rejects_unknown_camera():
    response = client.get("/api/cameras/UNKNOWN/feed-info")
    assert response.status_code == 404


def test_aicc_path_traversal_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    monkeypatch.setattr(main, "AICC_VIDEO_ENABLED", True)
    response = client.post("/api/aicc/predict-data-video", json={"path": "../../app/main.py"})
    assert response.status_code == 400


def test_aicc_job_unknown_id_returns_404():
    response = client.get("/api/aicc/jobs/not-a-real-job")
    assert response.status_code == 404


def test_camera_feed_can_be_assigned_from_data_folder(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(main, "PUBLIC_DEMO_MODE", False)
    data_dir = tmp_path / "data"
    feeds_dir = data_dir / "camera_feeds"
    data_dir.mkdir(parents=True)
    feeds_dir.mkdir()
    source = data_dir / "input.mp4"
    source.write_bytes(b"local-demo-clip-bytes")
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "CAMERA_FEED_DIR", feeds_dir)
    monkeypatch.setattr(main, "CAMERA_FEED_INDEX_PATH", feeds_dir / "index.json")
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: _FakeCapture())

    response = client.post("/api/cameras/HN-001/feed-from-data", json={"path": "input.mp4"})
    assert response.status_code == 200, response.text
    feed = response.json()["feed"]
    assert feed["available"] is True
    assert feed["source_type"] == "assigned_clip"
    assigned = feeds_dir / feed["filename"]
    assert assigned.read_bytes() == source.read_bytes()
    index = json.loads((feeds_dir / "index.json").read_text(encoding="utf-8"))
    assert index["HN-001"] == feed["filename"]


def test_shared_input_video_is_not_automatically_mislabeled_as_each_camera(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    feeds_dir = data_dir / "camera_feeds"
    feeds_dir.mkdir(parents=True)
    (data_dir / "input.mp4").write_bytes(b"global-sample")
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "CAMERA_FEED_DIR", feeds_dir)
    monkeypatch.setattr(main, "CAMERA_FEED_INDEX_PATH", feeds_dir / "index.json")
    info = main._camera_feed_info("HN-001")
    assert info["available"] is False
    assert info["source_type"] == "none"
