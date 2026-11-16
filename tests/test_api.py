import time

import numpy as np
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_home_and_status():
    assert "hanoiFlow" in client.get("/").text
    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["max_duration_seconds"] == main.MAX_VIDEO_SECONDS
    assert status["library_videos"] >= 1


def test_upload_rejects_unsupported_and_unreadable_files():
    assert client.post("/api/analyze", files={"file": ("a.txt", b"x")}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"not a video")}).status_code == 400
    assert client.get("/api/jobs/not-an-id").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404


def test_job_runs_and_returns_the_best_checked_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_check_video", lambda path: None)
    monkeypatch.setattr(main.detector, "load", lambda: None)
    seen = {}

    def fake_analyze(path, detector, **kwargs):
        seen["library"] = kwargs["library_videos"]
        series = {"t": np.arange(4.0), "ci": np.zeros(4), "occupancy": np.zeros(4), "count": np.zeros(4), "motion": np.zeros(4)}
        cam = lambda i, checked: {"id": i, "label": f"Camera {i + 1}", "on_air_seconds": 60.0, "intervals": [[0, 60]],
                                   "series": series, "checked_forecasts": checked,
                                   "minute_forecast": {"available": True, "camera": i}}
        return {"video": {"duration_seconds": 60}, "mode": "broadcast", "cameras": [cam(0, 3), cam(1, 9)], "timeline": []}

    monkeypatch.setattr(main, "analyze_video", fake_analyze)
    response = client.post("/api/analyze", files={"file": ("clip.mp4", b"\0" * 1024)}, data={"timing_confirmed": "true"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["main_camera"] == 1 and result["minute_forecast"]["camera"] == 1
    assert all("series" not in cam for cam in result["cameras"])
    assert seen["library"] >= 1
    assert not list((main.UPLOAD_DIR).glob("upload_*")), "uploaded video must be deleted after analysis"


def test_library_skips_records_cut_from_the_uploaded_video():
    files = main._library_files()
    series, videos = main.load_library(None)
    assert len(series) == len(files) and videos >= 1
    import json
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        source = (record.get("source_video_keys") or [record["video_key"]])[0]
        sharing = sum(1 for p in files if source in {json.loads(p.read_text(encoding="utf-8"))["video_key"],
                                                     *(json.loads(p.read_text(encoding="utf-8")).get("source_video_keys") or [])})
        assert len(main.load_library(source)[0]) == len(files) - sharing


def test_map_page_and_weather_endpoints(monkeypatch):
    assert "leaflet" in client.get("/map").text
    config = client.get("/api/map/config").json()
    assert config["tile_url_template"].startswith("https://") and config["boundary_url"].endswith(".geojson")
    assert client.get(config["boundary_url"]).status_code == 200
    assert client.get("/api/weather", params={"latitude": 10.77, "longitude": 106.70}).status_code == 422
    monkeypatch.setattr(main.weather, "forecast", lambda lat, lon: {"current": {"temperature_c": 24}})
    assert client.get("/api/weather").json()["current"]["temperature_c"] == 24

    def down():
        raise main.weather.WeatherError("offline")
    monkeypatch.setattr(main.weather, "overview", down)
    assert client.get("/api/weather/overview").status_code == 502


def test_routes_and_model_report(monkeypatch):
    config = client.get("/api/map/config").json()
    assert len(config["areas"]) >= 10
    assert client.get("/api/routes", params={"from": "10.77,106.70", "to": "21.03,105.85"}).status_code == 422
    assert client.get("/api/routes", params={"from": "abc", "to": "21.03,105.85"}).status_code == 422
    fake = [{"distance_m": 5000.0, "duration_seconds": 600.0, "geometry": {"type": "LineString", "coordinates": [[105.85, 21.03], [105.80, 21.04]]}}]
    monkeypatch.setattr(main.routing, "route_alternatives", lambda a, b: fake)
    body = client.get("/api/routes", params={"from": "21.0285,105.8542", "to": "21.0362,105.7906"}).json()
    assert body["routes"] == fake and "thời gian thực" in body["note"]
    report = client.get("/api/model/report").json()
    assert {s["horizon_minutes"] for s in report["summary"]} == {5, 10, 15}


def test_osrm_payload_is_parsed_fastest_first():
    payload = {"code": "Ok", "routes": [
        {"distance": 9000, "duration": 900, "geometry": {"coordinates": [[1, 2], [3, 4]]}},
        {"distance": 8000, "duration": 700, "geometry": {"coordinates": [[1, 2], [5, 6]]}},
        {"distance": 1, "duration": 1, "geometry": {"coordinates": [[1, 2]]}}]}
    routes = main.routing.parse_routes(payload)
    assert [r["duration_seconds"] for r in routes] == [700, 900]
    import pytest
    with pytest.raises(main.routing.RouteError):
        main.routing.parse_routes({"code": "NoRoute"})
