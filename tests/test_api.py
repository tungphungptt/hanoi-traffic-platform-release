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
    monkeypatch.setattr(main, "_check_video", lambda path: (640, 360))
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


def test_upload_token_and_hourly_limit(monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_TOKEN", "bi-mat")
    assert client.get("/api/status").json()["token_required"] is True
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"x")}).status_code == 401
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"x")}, headers={"X-Upload-Token": "sai"}).status_code == 401
    monkeypatch.setattr(main, "UPLOAD_TOKEN", "")
    monkeypatch.setattr(main, "UPLOADS_PER_HOUR", 1)
    monkeypatch.setattr(main, "_upload_times", [main._now()])
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"x")}).status_code == 429


def test_upload_guard_rejects_before_the_body_is_parsed(monkeypatch):
    """Bad token / oversize / busy answers come from the headers alone, nothing is spooled."""
    parsed = []
    real = main.analyze
    monkeypatch.setattr(main, "UPLOAD_TOKEN", "bi-mat")
    r = client.post("/api/analyze", content=b"x" * 10, headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 401 and not parsed
    monkeypatch.setattr(main, "UPLOAD_TOKEN", "")
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    assert client.post("/api/analyze", content=b"x" * (2 * 1024 * 1024),
                       headers={"content-type": "multipart/form-data; boundary=b"}).status_code == 413
    # No Content-Length (chunked): cut off once past the limit.
    def chunks():
        for _ in range(8):
            yield b"x" * (512 * 1024)
    r = client.post("/api/analyze", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert real is main.analyze


def test_demo_endpoint_lists_consecutive_segments():
    demo = client.get("/api/demo").json()
    starts = [s["start"] for s in demo["segments"]]
    assert starts == sorted(starts) and starts[0] == 0
    total = demo["segments"][-1]["start"] + demo["segments"][-1]["duration"]
    assert abs(total - demo["session"]["video"]["duration_seconds"]) < 5
    assert demo["session"]["contract"].startswith("hanoiflow-forecast-session/")


def test_live_intake_needs_a_token_and_serves_forecasts(monkeypatch):
    import time
    from traffic_brain.live import LiveStore
    monkeypatch.setattr(main, "LIVE", LiveStore(lambda: ([], 0), location_ok=main.is_hanoi_coordinate))
    now = time.time()
    body = {"name": "Thử", "lat": 21.0285, "lng": 105.8542,
            "samples": [{"t": now - 100 + i, "ci": 0.3, "vehicle_count": 5} for i in range(100)]}
    path = "/api/live/tdh-216/samples"
    monkeypatch.setattr(main, "LIVE_TOKEN", "")
    assert client.post(path, json=body).status_code == 503
    assert client.get("/api/live").json() == {"enabled": False, "cameras": []}
    monkeypatch.setattr(main, "LIVE_TOKEN", "thu-nhan")
    assert client.get("/api/status").json()["live_enabled"] is True
    assert client.post(path, json=body).status_code == 401
    assert client.post(path, json=body, headers={"X-Live-Token": "sai"}).status_code == 401
    ok = client.post(path, json=body, headers={"X-Live-Token": "thu-nhan"})
    assert ok.status_code == 200 and ok.json()["accepted"] == 100
    listing = client.get("/api/live").json()
    assert listing["enabled"] and [c["camera_id"] for c in listing["cameras"]] == ["tdh-216"]
    one = client.get("/api/live/tdh-216").json()
    assert one["status"] == "warming_up" and one["name"] == "Thử"
    assert client.get("/api/live/khong-co").status_code == 404
    outside = {**body, "lat": 10.0, "lng": 105.0}
    assert client.post(path, json=outside, headers={"X-Live-Token": "thu-nhan"}).status_code == 422
    big = {"samples": [{"t": now, "ci": 0.3, "pad": "x" * 100}] * 4000}
    assert client.post(path, json=big, headers={"X-Live-Token": "thu-nhan"}).status_code == 413
    assert client.delete("/api/live/tdh-216").status_code == 401
    assert client.delete("/api/live/tdh-216", headers={"X-Live-Token": "thu-nhan"}).status_code == 200


def test_live_export_is_token_gated_and_needs_a_history_file(monkeypatch, tmp_path):
    import time
    from traffic_brain.history import History
    from traffic_brain.live import LiveStore
    monkeypatch.setattr(main, "LIVE_TOKEN", "thu-nhan")
    monkeypatch.setattr(main, "HISTORY", None)
    assert client.get("/api/live/export").status_code == 401
    assert client.get("/api/live/export", headers={"X-Live-Token": "thu-nhan"}).status_code == 404
    history = History(tmp_path / "h.sqlite")
    monkeypatch.setattr(main, "HISTORY", history)
    monkeypatch.setattr(main, "LIVE", LiveStore(lambda: ([], 0), history=history))
    now = time.time()
    body = {"samples": [{"t": now - 20 + i, "ci": 0.3, "vehicle_count": 5} for i in range(20)]}
    assert client.post("/api/live/cam-a/samples", json=body, headers={"X-Live-Token": "thu-nhan"}).status_code == 200
    assert client.get("/api/live/history").json()["samples"] == 20
    out = client.get("/api/live/export?camera_id=cam-a", headers={"X-Live-Token": "thu-nhan"})
    assert out.status_code == 200 and out.headers["content-type"].startswith("text/csv")
    assert len(out.text.strip().splitlines()) == 21
    assert client.get("/api/live/export?limit=0", headers={"X-Live-Token": "thu-nhan"}).status_code == 422
    # "export" and "history" are routes, not camera ids.
    assert client.post("/api/live/export/samples", json=body, headers={"X-Live-Token": "thu-nhan"}).status_code == 422


def _fake_analysis(mode="single_camera"):
    series = {"t": np.arange(4.0), "ci": np.zeros(4), "occupancy": np.zeros(4), "count": np.zeros(4), "motion": np.zeros(4)}
    cam = lambda i, intervals: {"id": i, "label": f"Camera {i + 1}", "on_air_seconds": 60.0, "intervals": intervals,
                                 "series": series, "checked_forecasts": 3, "minute_forecast": {"available": True}}
    cams = [cam(0, [[0, 30]]), cam(1, [[40, 60]])] if mode == "broadcast" else [cam(0, [[0, 60]])]
    return {"video": {"duration_seconds": 60, "width": 640, "height": 360}, "mode": mode, "cameras": cams, "timeline": []}


GOOD_CALIBRATION = '{"points": [[0.3, 0.9], [0.7, 0.9], [0.58, 0.4], [0.44, 0.4]], "width_m": 7, "length_m": 40, "time_seconds": %s}'


def _run_analysis(monkeypatch, mode, calibration, speed_out=None):
    monkeypatch.setattr(main, "_check_video", lambda path: (640, 360))
    monkeypatch.setattr(main.detector, "load", lambda: None)
    monkeypatch.setattr(main, "analyze_video", lambda path, detector, **kw: _fake_analysis(mode))
    calls = []

    def fake_bursts(path, detector, plane, **kw):
        calls.append(kw["intervals"])
        return speed_out or {"bursts": [{"t_start": 0.0, "t_end": 12.0, "vehicles": 4, "median_kmh": 31.0, "mean_kmh": 30.0,
                                         "p85_kmh": 38.0, "stopped_share": 0.0}],
                             "overall": {"vehicles": 4, "median_kmh": 31.0, "mean_kmh": 30.0, "p85_kmh": 38.0, "stopped_share": 0.0},
                             "frames_processed": 49, "burst_seconds": 12.0, "warning": None}

    monkeypatch.setattr(main, "measure_bursts", fake_bursts)
    response = client.post("/api/analyze", files={"file": ("clip.mp4", b"\0" * 1024)},
                           data={"timing_confirmed": "true", "calibration": calibration})
    return response, calls


def _wait(job_id):
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    return job


def test_speed_is_measured_when_a_valid_calibration_is_sent(monkeypatch):
    response, calls = _run_analysis(monkeypatch, "single_camera", GOOD_CALIBRATION % 5)
    assert response.status_code == 202
    job = _wait(response.json()["job_id"])
    speed = job["result"]["speed"]
    assert speed["status"] == "ok" and speed["overall"]["median_kmh"] == 31.0 and speed["camera_id"] == 0
    assert speed["calibration"] == {"width_m": 7.0, "length_m": 40.0, "time_seconds": 5.0}
    assert calls == [[[0, 60]]]


def test_speed_follows_the_camera_on_air_at_the_calibration_time(monkeypatch):
    response, calls = _run_analysis(monkeypatch, "broadcast", GOOD_CALIBRATION % 50)
    job = _wait(response.json()["job_id"])
    assert job["result"]["speed"]["camera_id"] == 1 and calls == [[[40, 60]]]
    response, calls = _run_analysis(monkeypatch, "broadcast", GOOD_CALIBRATION % 35)  # between the two cameras: nobody on air
    job = _wait(response.json()["job_id"])
    assert job["result"]["speed"]["status"] == "skipped" and calls == []


def test_no_calibration_means_no_speed_pass_and_a_bad_one_is_refused_up_front(monkeypatch):
    response, calls = _run_analysis(monkeypatch, "single_camera", "")
    job = _wait(response.json()["job_id"])
    assert "speed" not in job["result"] and calls == []
    bad, calls = _run_analysis(monkeypatch, "single_camera", '{"points": [[0, 0]], "width_m": 7, "length_m": 40}')
    assert bad.status_code == 422 and "Calibration" in bad.json()["detail"] and calls == []
    assert not list(main.UPLOAD_DIR.glob("upload_*")), "a refused upload must not leave its video behind"
