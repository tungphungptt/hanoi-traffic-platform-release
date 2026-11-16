import time

import numpy as np

from traffic_brain.history import History
from traffic_brain.live import LiveStore


def _store(path, clock, weather=None):
    history = History(path)
    return LiveStore(lambda: ([], 0), clock=lambda: clock[0], history=history, weather_provider=weather,
                     location_ok=lambda lat, lng: True), history


def _rows(t0, n):
    return [{"t": t0 + i, "ci": 0.3 + 0.001 * i, "vehicle_count": 5, "occupancy_ratio": 0.1} for i in range(n)]


def test_measurements_survive_a_restart(tmp_path):
    clock = [2e9]
    store, history = _store(tmp_path / "h.sqlite", clock)
    store.observe("tdh", _rows(clock[0] - 300, 300), name="TDH", lat=21.0, lng=105.8)
    history.close()
    # New process, same file: the camera and its recent samples come back, nothing is re-sent.
    clock2 = [clock[0] + 60]
    again, history2 = _store(tmp_path / "h.sqlite", clock2)
    assert again.restore() == 300
    assert again.camera_ids() == ["tdh"]
    status = again.forecast("tdh")
    assert status["name"] == "TDH" and status["samples"] == 300 and status["lat"] == 21.0
    # Writing the same timestamps again does not duplicate rows.
    again.observe("tdh", _rows(clock[0] - 300, 300))
    assert history2.counts()["samples"] == 300


def test_old_samples_in_the_file_are_not_lost_when_the_server_was_down(tmp_path):
    clock = [2e9]
    store, history = _store(tmp_path / "h.sqlite", clock)
    store.observe("a", _rows(clock[0] - 100, 100))
    history.close()
    later, _ = _store(tmp_path / "h.sqlite", [clock[0] + 5 * 86400])  # five days later
    assert later.restore() == 100


def test_weather_is_snapshotted_off_the_request_path_and_joined_in_the_export(tmp_path):
    clock = [2e9]
    calls = []

    def provider(lat, lng):
        calls.append((lat, lng))
        return {"temperature_c": 31.5, "humidity_pct": 70, "precipitation_mm": 1.2, "wind_kmh": 8, "weather_code": 61}

    store, history = _store(tmp_path / "h.sqlite", clock, provider)
    store.observe("tdh", _rows(clock[0] - 60, 60), lat=21.0, lng=105.8)
    deadline = time.time() + 3
    while history.counts()["weather_snapshots"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert calls == [(21.0, 105.8)]
    # A second post within ten minutes does not call the weather service again.
    clock[0] += 30
    store.observe("tdh", _rows(clock[0], 10))
    time.sleep(0.1)
    assert len(calls) == 1
    lines = history.export_csv("tdh").strip().splitlines()
    header = lines[0].split(",")
    assert header[:4] == ["camera_id", "t_epoch", "t_local", "ci"] and "precipitation_mm" in header
    first = dict(zip(header, lines[1].split(",")))
    # The snapshot was taken at clock[0] (2e9), the first sample is 60 s earlier: no weather yet (never look ahead).
    assert first["precipitation_mm"] == ""
    last = dict(zip(header, lines[-1].split(",")))
    assert last["precipitation_mm"] == "1.2" and last["weather_code"] == "61"
    assert len(lines) - 1 == 70


def test_weather_failure_leaves_a_gap_and_does_not_break_intake(tmp_path):
    clock = [2e9]

    def boom(lat, lng):
        raise RuntimeError("offline")

    store, history = _store(tmp_path / "h.sqlite", clock, boom)
    assert store.observe("a", _rows(clock[0] - 10, 10), lat=21.0, lng=105.8)["accepted"] == 10
    time.sleep(0.1)
    assert history.counts()["weather_snapshots"] == 0 and history.counts()["samples"] == 10


def test_local_time_column_uses_hanoi_clock(tmp_path):
    history = History(tmp_path / "h.sqlite")
    history.add_samples("a", [{"t": 1_700_000_000.0, "raw_ci": 0.5}])
    row = history.export_csv().strip().splitlines()[1].split(",")
    assert row[2] == "2023-11-15 05:13:20"  # 2023-11-14 22:13:20 UTC + 7 h
