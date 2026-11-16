"""Live congestion forecasts from an authenticated collector.

A collector (``tools/live_collector.py``) reads an authorised camera stream, measures it with
``video_analysis.FrameMeter`` and posts the 1 Hz measurements here. The store keeps a bounded
history per camera in memory (lost on restart) and, on request, runs the same causal forecaster
used for uploaded videos on that history. Nothing here opens a camera.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable, Mapping

from traffic_brain.history import History
from traffic_brain.minute_forecaster import (
    GRID_STEP_SECONDS, HORIZONS_MINUTES, WARMUP_SECONDS, build_minute_forecast,
)
from traffic_brain.traffic_levels import level_for_ci

CAMERA_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
MAX_SAMPLES = 3 * 3600        # per camera: 3 hours at 1 Hz
MAX_CAMERAS = 64
MAX_BATCH = 600
MAX_AGE_SECONDS = 900         # a sample older than this is rejected (stale or replayed)
FUTURE_SLACK_SECONDS = 15
STALE_SECONDS = 120           # no new sample for this long: the forecast is no longer "now"
CACHE_SECONDS = 10
WEATHER_EVERY_SECONDS = 600   # one weather snapshot per camera every 10 minutes
RECENT_SECONDS = 3600
FIELDS = ("occupancy_ratio", "vehicle_count", "normalized_speed", "turnover")


RESERVED_IDS = frozenset({"export", "history"})  # their own routes under /api/live/


class LiveError(ValueError):
    """Invalid camera id, sample or location."""


def _number(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LiveError(f"{name} phải là số hữu hạn")
    if not lo <= value <= hi:
        raise LiveError(f"{name} phải nằm trong [{lo}, {hi}]")
    return float(value)


class LiveStore:
    def __init__(self, library_loader: Callable[[], tuple[list, int]], *,
                 clock: Callable[[], float] = time.time,
                 location_ok: Callable[[float, float], bool] | None = None,
                 history: History | None = None,
                 weather_provider: Callable[[float, float], Mapping[str, Any] | None] | None = None):
        self._history = history
        self._weather_provider = weather_provider
        self._library_loader = library_loader
        self._clock = clock
        self._location_ok = location_ok
        self._lock = threading.RLock()
        self._cameras: dict[str, dict[str, Any]] = {}

    def restore(self) -> int:
        """Reload the last hours of every camera from the history file after a restart."""
        if self._history is None:
            return 0
        restored = 0
        now = self._clock()
        for meta in self._history.cameras():
            rows = self._history.recent(meta["camera_id"], MAX_SAMPLES, now)
            if not rows:
                continue
            with self._lock:
                self._cameras[meta["camera_id"]] = {
                    "samples": deque(rows, maxlen=MAX_SAMPLES), "name": meta["name"] or meta["camera_id"],
                    "lat": meta["lat"], "lng": meta["lng"], "cache": None, "weather_at": -math.inf}
            restored += len(rows)
        return restored

    # ------------------------------------------------------------ ingest

    def observe(self, camera_id: str, samples: Iterable[Mapping[str, Any]], *,
                name: str | None = None, lat: float | None = None, lng: float | None = None) -> dict[str, Any]:
        if not isinstance(camera_id, str) or not CAMERA_ID.fullmatch(camera_id) or camera_id in RESERVED_IDS:
            raise LiveError("Mã camera gồm chữ thường, số, '-' hoặc '_' (tối đa 40 ký tự).")
        samples = list(samples)
        if not samples:
            raise LiveError("Không có mẫu đo nào.")
        if len(samples) > MAX_BATCH:
            raise LiveError(f"Mỗi lần gửi tối đa {MAX_BATCH} mẫu.")
        if (lat is None) != (lng is None):
            raise LiveError("Cần gửi cả lat và lng.")
        if lat is not None:
            lat, lng = _number(lat, "lat", -90, 90), _number(lng, "lng", -180, 180)
            if self._location_ok and not self._location_ok(lat, lng):
                raise LiveError("Tọa độ camera nằm ngoài ranh giới Hà Nội.")
        now = self._clock()
        clean = []
        for item in samples:
            if not isinstance(item, Mapping):
                raise LiveError("Mỗi mẫu phải là một đối tượng.")
            t = _number(item.get("t"), "t", 1e9, 1e11)
            if t > now + FUTURE_SLACK_SECONDS:
                raise LiveError("Mẫu có thời điểm trong tương lai.")
            if now - t > MAX_AGE_SECONDS:
                raise LiveError(f"Mẫu cũ hơn {MAX_AGE_SECONDS // 60} phút bị từ chối.")
            row = {"t": t, "raw_ci": _number(item.get("ci"), "ci", 0.0, 1.0)}
            for key in FIELDS:
                value = item.get(key)
                if value is not None:
                    row[key] = _number(value, key, 0.0, 1e6)
            clean.append(row)
        clean.sort(key=lambda r: r["t"])
        with self._lock:
            if camera_id not in self._cameras and len(self._cameras) >= MAX_CAMERAS:
                raise LiveError(f"Tối đa {MAX_CAMERAS} camera.")
            cam = self._cameras.setdefault(camera_id, {
                "samples": deque(maxlen=MAX_SAMPLES), "name": camera_id, "lat": None, "lng": None, "cache": None,
                "weather_at": -math.inf})
            if name:
                cam["name"] = str(name).strip()[:80] or camera_id
            if lat is not None:
                cam["lat"], cam["lng"] = lat, lng
            last = cam["samples"][-1]["t"] if cam["samples"] else -math.inf
            kept = []
            for row in clean:
                if row["t"] > last:
                    cam["samples"].append(row)
                    kept.append(row)
                    last = row["t"]
            accepted = len(kept)
            if accepted:
                cam["cache"] = None
            result = {"camera_id": camera_id, "accepted": accepted, "ignored": len(clean) - accepted,
                      "stored": len(cam["samples"])}
            snapshot_due = (self._weather_provider is not None and cam["lat"] is not None
                            and now - cam["weather_at"] >= WEATHER_EVERY_SECONDS)
            if snapshot_due:
                cam["weather_at"] = now
            meta = (cam["name"], cam["lat"], cam["lng"])
        if self._history is not None:
            self._history.save_camera(camera_id, *meta)
            self._history.add_samples(camera_id, kept)
        if snapshot_due:
            threading.Thread(target=self._snapshot_weather, args=(camera_id, meta[1], meta[2], now), daemon=True).start()
        return result

    def _snapshot_weather(self, camera_id: str, lat: float, lng: float, t: float) -> None:
        """Weather next to the measurements, fetched off the request path; a failure just leaves a gap."""
        if self._history is None or self._weather_provider is None:
            return
        try:
            snapshot = self._weather_provider(lat, lng)
        except Exception:
            return
        if snapshot:
            self._history.add_weather(camera_id, t, snapshot)

    def remove(self, camera_id: str) -> bool:
        with self._lock:
            return self._cameras.pop(camera_id, None) is not None

    # ------------------------------------------------------------ read

    def camera_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._cameras)

    def forecast(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam is None:
                return None
            now = self._clock()
            cached = cam["cache"]
            if cached and now - cached[0] < CACHE_SECONDS:
                return cached[1]
            samples = list(cam["samples"])
            meta = {"camera_id": camera_id, "name": cam["name"], "lat": cam["lat"], "lng": cam["lng"]}
        result = {**meta, **self._compute(samples, now)}
        with self._lock:
            if camera_id in self._cameras:
                self._cameras[camera_id]["cache"] = (now, result)
        return result

    def _compute(self, samples: list[dict[str, Any]], now: float) -> dict[str, Any]:
        if not samples:
            return {"status": "warming_up", "samples": 0, "message": "Chưa nhận mẫu đo nào."}
        origin, last = samples[0]["t"], samples[-1]["t"]
        base = {"samples": len(samples), "span_seconds": round(last - origin, 1),
                "updated_at": last, "age_seconds": round(max(0.0, now - last), 1)}
        needed = WARMUP_SECONDS + HORIZONS_MINUTES[0] * 60.0 + 4 * GRID_STEP_SECONDS
        if last - origin < needed:
            return {**base, "status": "warming_up", "needed_seconds": round(needed),
                    "message": f"Cần khoảng {math.ceil(needed / 60)} phút số liệu liên tục để dự báo +5 phút."}
        timeline = [{**{k: v for k, v in s.items() if k != "t"}, "timestamp_seconds": s["t"] - origin} for s in samples]
        library, library_videos = self._library_loader()
        forecast = build_minute_forecast(timeline, duration_seconds=last - origin, timing_confirmed=True,
                                         library=library, library_videos=library_videos)
        latest = forecast.get("latest")
        if not forecast.get("available") or not latest:
            return {**base, "status": "warming_up", "message": forecast.get("message", "Chưa đủ số liệu hợp lệ.")}
        anchor_epoch = origin + latest["t"]
        horizons = [{
            "horizon_minutes": h["horizon_minutes"], "target_epoch": origin + h["target_t"],
            "ci": h["ci"], "lower": h["lower"], "upper": h["upper"], "level_key": h["level_key"],
            "p_gridlock": h["p_gridlock"], "warning": h["warning"], "watch": h["watch"],
        } for h in latest["horizons"]]
        recent = [{"epoch": origin + o["t"], "ci": o["ci"]} for o in forecast["observed"]
                  if o["ci"] is not None and origin + o["t"] >= last - RECENT_SECONDS]
        checked = [{"horizon_minutes": m["horizon_minutes"], "evaluated": m.get("evaluated", 0),
                    "skill_vs_persistence": m.get("skill_vs_persistence")} for m in forecast["metrics"]]
        stale = base["age_seconds"] > STALE_SECONDS
        return {**base, "status": "stale" if stale else "live", "forecast_epoch": anchor_epoch,
                "observed_ci": latest["observed_ci"], "level_key": level_for_ci(latest["observed_ci"])[0],
                "horizons": horizons, "recent": recent, "checked": checked,
                "library_videos": forecast["method"]["library_videos"],
                "thresholds": forecast["thresholds"], "state_labels": forecast["state_labels"]}
