import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from traffic_brain.live import LiveError, LiveStore, MAX_BATCH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_collector  # noqa: E402


def _store(clock):
    return LiveStore(lambda: ([], 0), clock=lambda: clock[0], location_ok=lambda lat, lng: 20.5 < lat < 21.5)


def _rows(t0, n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"t": t0 + i, "ci": float(np.clip(0.3 + 0.15 * np.sin(i / 150) + rng.normal(0, 0.03), 0, 1)),
             "vehicle_count": 8} for i in range(n)]


def test_rejects_bad_samples_ids_and_places():
    clock = [2e9]
    store = _store(clock)
    ok = {"t": clock[0], "ci": 0.4}
    with pytest.raises(LiveError):
        store.observe("Bad Id", [ok])
    for bad in ({"t": clock[0] + 100, "ci": 0.4}, {"t": clock[0] - 2000, "ci": 0.4}, {"t": clock[0], "ci": 1.5},
                {"t": clock[0], "ci": float("nan")}, {"t": clock[0], "ci": True}, {"t": clock[0], "ci": 0.4, "vehicle_count": -1}):
        with pytest.raises(LiveError):
            store.observe("cam-1", [bad])
    with pytest.raises(LiveError):
        store.observe("cam-1", [ok] * (MAX_BATCH + 1))
    with pytest.raises(LiveError):
        store.observe("cam-1", [ok], lat=10.0, lng=105.0)  # outside Hanoi
    with pytest.raises(LiveError):
        store.observe("cam-1", [ok], lat=21.0)             # lng missing
    assert store.camera_ids() == []


def test_old_or_repeated_samples_are_ignored_and_forecast_needs_history():
    clock = [2e9]
    store = _store(clock)
    assert store.observe("cam-1", _rows(clock[0] - 100, 100), name="Thử", lat=21.0, lng=105.8)["accepted"] == 100
    again = store.observe("cam-1", _rows(clock[0] - 100, 100))
    assert again["accepted"] == 0 and again["ignored"] == 100
    result = store.forecast("cam-1")
    assert result["status"] == "warming_up" and result["name"] == "Thử" and result["lat"] == 21.0
    assert store.forecast("nope") is None


def test_forecast_after_enough_history_then_goes_stale():
    clock = [2e9]
    store = _store(clock)
    t0 = clock[0]
    rows = _rows(t0, 1500)
    for i in range(0, 1500, 500):
        clock[0] = t0 + i + 500
        store.observe("cam-1", rows[i:i + 500])
    result = store.forecast("cam-1")
    assert result["status"] == "live" and result["age_seconds"] < 5
    assert [h["horizon_minutes"] for h in result["horizons"]] == [5, 10, 15]
    for h in result["horizons"]:
        assert 0 <= h["lower"] <= h["ci"] <= h["upper"] <= 1 and 0 <= h["p_gridlock"] <= 1
        assert h["target_epoch"] > result["forecast_epoch"]
    assert result["recent"] and result["recent"][-1]["epoch"] <= clock[0]
    assert store.forecast("cam-1") is result  # cached for a few seconds
    clock[0] += 400
    store._cameras["cam-1"]["cache"] = None
    assert store.forecast("cam-1")["status"] == "stale"
    assert store.remove("cam-1") and store.forecast("cam-1") is None


def test_collector_buffers_while_the_server_is_down_and_stops_on_a_bad_token():
    ticks = iter(range(0, 1000))
    samples = [{"t": 1e9 + i, "ci": 0.3} for i in range(12)]
    posts, fail = [], {"on": True}

    def post(payload):
        if fail["on"]:
            raise urllib.error.URLError("down")
        posts.append(payload)

    def clock():
        return next(ticks) * 1.0

    # Server down for the first batches, back up later: nothing is lost (within the buffer).
    def gen():
        for i, s in enumerate(samples):
            if i == 8:
                fail["on"] = False
            yield s
    sent = live_collector.run(gen(), post, meta={"name": "x"}, batch_seconds=3, clock=clock)
    assert sent == 12 and sum(len(p["samples"]) for p in posts) >= 12 - 0
    assert posts[0]["samples"][0]["t"] == 1e9 and posts[0]["name"] == "x"

    def denied(payload):
        raise urllib.error.HTTPError("u", 401, "no", {}, None)
    with pytest.raises(SystemExit):
        live_collector.run(iter(samples), denied, meta={}, batch_seconds=0, clock=clock)


def test_measure_stream_on_a_video_file_paced_like_real_time(tmp_path):
    import cv2
    pytest.importorskip("onnxruntime")
    from traffic_brain.detector import Detector
    path = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 180))
    rng = np.random.default_rng(0)
    background = rng.integers(60, 120, (180, 320, 3), dtype=np.uint8)
    for i in range(40):
        frame = background.copy()
        cv2.rectangle(frame, (10 + i * 5, 80), (60 + i * 5, 110), (30, 30, 200), -1)
        writer.write(frame)
    writer.release()
    now = [1000.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    out = list(live_collector.measure_stream(str(path), Detector(), clock=lambda: now[0], sleep=sleep))
    assert 18 <= len(out) <= 21  # 20 s of video at 1 sample per second
    times = [s["t"] for s in out]
    assert all(b - a == pytest.approx(1.0, abs=0.01) for a, b in zip(times, times[1:]))
    assert all(0 <= s["ci"] <= 1 and "vehicle_count" in s for s in out)
    json.dumps(out)  # plain JSON types only
    assert sum(slept) == pytest.approx(19, abs=1.5)
