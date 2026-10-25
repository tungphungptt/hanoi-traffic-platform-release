from __future__ import annotations

"""Bounded-memory traffic analytics primitives (independently implemented for this project).

The functions consume detector/tracker observations; they do not open cameras, run YOLO,
or claim that an uncalibrated pixel displacement is km/h. Coordinates are normalized
([0, 1]) so each camera can have its own geometry and calibration.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable, Mapping, Sequence


class AnalyticsInputError(ValueError):
    """Invalid tracking observation or line configuration."""


def _num(value: Any, name: str, *, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool):
        raise AnalyticsInputError(f"{name} phải là số hữu hạn")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalyticsInputError(f"{name} phải là số hữu hạn") from exc
    if not math.isfinite(result):
        raise AnalyticsInputError(f"{name} phải là số hữu hạn")
    if lo is not None and result < lo:
        raise AnalyticsInputError(f"{name} phải >= {lo}")
    if hi is not None and result > hi:
        raise AnalyticsInputError(f"{name} phải <= {hi}")
    return result


def adaptive_frame_stride(source_fps: float, target_fps: float = 6.0, *, max_target_fps: float = 12.0) -> dict[str, float | int]:
    """Return a deterministic frame stride/sample rate for CPU-budgeted inference."""
    fps = _num(source_fps, "source_fps", lo=0.001, hi=1000.0)
    target = _num(target_fps, "target_fps", lo=0.1, hi=120.0)
    maximum = _num(max_target_fps, "max_target_fps", lo=0.1, hi=120.0)
    target = min(target, maximum, fps)
    stride = max(1, int(round(fps / target)))
    return {"frame_stride": stride, "sample_fps": round(fps / stride, 6), "source_fps": fps}


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float], eps: float = 1e-9) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    """Inclusive 2D segment intersection, including a shared endpoint."""
    aa = (_num(a[0], "a.x"), _num(a[1], "a.y"))
    bb = (_num(b[0], "b.x"), _num(b[1], "b.y"))
    cc = (_num(c[0], "c.x"), _num(c[1], "c.y"))
    dd = (_num(d[0], "d.x"), _num(d[1], "d.y"))
    o1, o2 = _orientation(aa, bb, cc), _orientation(aa, bb, dd)
    o3, o4 = _orientation(cc, dd, aa), _orientation(cc, dd, bb)
    if ((o1 > 0 > o2 or o2 > 0 > o1) and (o3 > 0 > o4 or o4 > 0 > o3)):
        return True
    eps = 1e-9
    return ((abs(o1) <= eps and _on_segment(aa, bb, cc)) or
            (abs(o2) <= eps and _on_segment(aa, bb, dd)) or
            (abs(o3) <= eps and _on_segment(cc, dd, aa)) or
            (abs(o4) <= eps and _on_segment(cc, dd, bb)))


@dataclass
class _TrackState:
    x: float
    y: float
    first_seen: float
    last_seen: float
    frame_count: int = 1
    path_length: float = 0.0
    stationary_since: float | None = None
    stationary_reported: bool = False
    last_class: str = "unknown"


@dataclass
class TrafficAnalyticsEngine:
    """Per-camera count/event aggregator with bounded queues and duplicate-safe line counts.

    Heuristics here are *candidate events* for operator review, never autonomous control
    commands. Supply one entry per track in each sampled frame, with normalized x/y.
    """
    history_size: int = 120
    event_size: int = 500
    motion_threshold: float = 0.004
    stationary_threshold: float = 0.0015
    stationary_seconds: float = 10.0
    surge_min_samples: int = 10
    surge_sigma: float = 2.0
    confidence_drift_ratio: float = 0.70
    max_track_states: int = 10000
    max_counted_keys: int = 50000
    max_line_counters: int = 200
    _tracks: dict[str, _TrackState] = field(default_factory=dict, init=False)
    _counted: OrderedDict[tuple[str, str], None] = field(default_factory=OrderedDict, init=False)
    _counts_by_line: Counter[str] = field(default_factory=Counter, init=False)
    _active_history: deque[int] = field(default_factory=deque, init=False)
    _confidence_history: deque[float] = field(default_factory=deque, init=False)
    _events: deque[dict[str, Any]] = field(default_factory=deque, init=False)
    _last_event_time: dict[str, float] = field(default_factory=dict, init=False)
    _low_confidence_streak: int = field(default=0, init=False)
    _last_frame_epoch: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.history_size < 10 or self.event_size < 1:
            raise AnalyticsInputError("history_size tối thiểu 10 và event_size tối thiểu 1")
        if self.max_track_states < 1 or self.max_counted_keys < 1 or self.max_line_counters < 1:
            raise AnalyticsInputError("Các giới hạn bộ nhớ phải lớn hơn 0")
        self._active_history = deque(maxlen=min(int(self.history_size), 600))
        self._confidence_history = deque(maxlen=min(int(self.history_size), 600))
        self._events = deque(maxlen=min(int(self.event_size), 2000))

    def _emit(self, event_type: str, now: float, details: Mapping[str, Any], cooldown: float = 30.0) -> dict[str, Any] | None:
        if now - self._last_event_time.get(event_type, -1e12) < cooldown:
            return None
        event = {"event_type": event_type, "timestamp_epoch": round(now, 3), "details": dict(details), "operator_review_required": True}
        self._last_event_time[event_type] = now
        self._events.append(event)
        return event

    def process_frame(
        self,
        tracks: Iterable[Mapping[str, Any]],
        *,
        timestamp_epoch: float | None = None,
        counting_lines: Iterable[Mapping[str, Any]] = (),
        detector_confidence: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if timestamp_epoch is None else _num(timestamp_epoch, "timestamp_epoch", lo=0)
        if self._last_frame_epoch is not None and now <= self._last_frame_epoch:
            raise AnalyticsInputError("timestamp_epoch phải tăng nghiêm ngặt theo thứ tự frame")
        self._last_frame_epoch = now
        cleaned: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for i, raw in enumerate(tracks):
            track_id = str(raw.get("track_id", "")).strip()
            if not track_id or len(track_id) > 128:
                raise AnalyticsInputError(f"tracks[{i}].track_id phải có độ dài 1–128")
            if track_id in seen_ids:
                raise AnalyticsInputError(f"track_id '{track_id}' bị lặp trong cùng một frame")
            seen_ids.add(track_id)
            class_name = str(raw.get("class_name", "unknown")).strip().lower()
            if not class_name or len(class_name) > 64:
                raise AnalyticsInputError(f"tracks[{i}].class_name không hợp lệ")
            x = _num(raw.get("x"), f"tracks[{i}].x", lo=0, hi=1)
            y = _num(raw.get("y"), f"tracks[{i}].y", lo=0, hi=1)
            confidence = raw.get("confidence")
            confidence_value = None if confidence is None else _num(confidence, f"tracks[{i}].confidence", lo=0, hi=1)
            cleaned.append({"track_id": track_id, "class_name": class_name, "x": x, "y": y, "confidence": confidence_value})

        lines: list[dict[str, Any]] = []
        line_ids: set[str] = set()
        for i, line in enumerate(counting_lines):
            line_id = str(line.get("line_id", "")).strip()
            if not line_id or len(line_id) > 64 or line_id in line_ids:
                raise AnalyticsInputError(f"counting_lines[{i}].line_id bị thiếu, quá dài hoặc bị lặp")
            line_ids.add(line_id)
            a = (_num(line.get("x1"), f"{line_id}.x1", lo=0, hi=1), _num(line.get("y1"), f"{line_id}.y1", lo=0, hi=1))
            b = (_num(line.get("x2"), f"{line_id}.x2", lo=0, hi=1), _num(line.get("y2"), f"{line_id}.y2", lo=0, hi=1))
            if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6:
                raise AnalyticsInputError(f"Đường đếm '{line_id}' có hai đầu mút trùng nhau")
            lines.append({"line_id": line_id, "a": a, "b": b})

        crossing_events: list[dict[str, Any]] = []
        stationary_events: list[dict[str, Any]] = []
        for item in cleaned:
            tid, xy = item["track_id"], (item["x"], item["y"])
            state = self._tracks.get(tid)
            if state is None or now - state.last_seen > 5.0:
                # Long gaps break geometric continuity: do not fabricate line crossings across an occlusion.
                self._tracks[tid] = _TrackState(xy[0], xy[1], now, now, last_class=item["class_name"])
                continue
            # Replayed or out-of-order timestamps should not corrupt velocity/stationary calculations.
            if now < state.last_seen:
                continue
            distance = math.hypot(xy[0] - state.x, xy[1] - state.y)
            state.path_length += distance
            state.frame_count += 1
            state.last_class = item["class_name"]
            if distance <= self.stationary_threshold:
                if state.stationary_since is None:
                    state.stationary_since = state.last_seen
                if (now - state.stationary_since >= self.stationary_seconds and state.frame_count >= 4
                        and not state.stationary_reported):
                    event = self._emit("stationary_vehicle_candidate", now, {"track_id": tid, "class_name": item["class_name"]}, cooldown=60)
                    if event:
                        stationary_events.append(event)
                    state.stationary_reported = True
            else:
                state.stationary_since = None
                state.stationary_reported = False
            if state.path_length >= self.motion_threshold:
                for line in lines:
                    key = (tid, line["line_id"])
                    if key not in self._counted and segments_intersect((state.x, state.y), xy, line["a"], line["b"]):
                        self._counted[key] = None
                        self._counted.move_to_end(key)
                        while len(self._counted) > self.max_counted_keys:
                            self._counted.popitem(last=False)
                        if line["line_id"] not in self._counts_by_line and len(self._counts_by_line) >= self.max_line_counters:
                            oldest_line = next(iter(self._counts_by_line))
                            del self._counts_by_line[oldest_line]
                        self._counts_by_line[line["line_id"]] += 1
                        evt = {"event_type": "counting_line_crossing", "timestamp_epoch": round(now, 3),
                               "details": {"track_id": tid, "class_name": item["class_name"], "line_id": line["line_id"], "count": self._counts_by_line[line["line_id"]]},
                               "operator_review_required": False}
                        self._events.append(evt)
                        crossing_events.append(evt)
            state.x, state.y, state.last_seen = xy[0], xy[1], now

        # Bound state even if upstream gives new IDs forever. Oldest unseen tracks are pruned.
        stale_after = 120.0
        stale_ids = [tid for tid, state in self._tracks.items() if now - state.last_seen > stale_after]
        for tid in stale_ids:
            del self._tracks[tid]
        if len(self._tracks) > self.max_track_states:
            ordered = sorted(self._tracks.items(), key=lambda entry: entry[1].last_seen)
            evicted = [tid for tid, _ in ordered[:len(self._tracks) - self.max_track_states]]
            for tid in evicted:
                del self._tracks[tid]
            stale_ids.extend(evicted)
        if stale_ids:
            stale_set = set(stale_ids)
            for key in list(self._counted):
                if key[0] in stale_set:
                    del self._counted[key]

        class_counts = dict(sorted(Counter(item["class_name"] for item in cleaned).items()))
        active_count = len(cleaned)
        prior_counts = list(self._active_history)
        event_items: list[dict[str, Any]] = crossing_events + stationary_events
        if len(prior_counts) >= self.surge_min_samples:
            avg = sum(prior_counts) / len(prior_counts)
            std = math.sqrt(sum((value - avg) ** 2 for value in prior_counts) / len(prior_counts))
            if active_count > avg + max(self.surge_sigma * std, 3.0):
                event = self._emit("traffic_volume_surge_candidate", now, {"active_tracks": active_count, "rolling_mean": round(avg, 3), "rolling_std": round(std, 3)})
                if event:
                    event_items.append(event)
        self._active_history.append(active_count)

        confidence_values = [item["confidence"] for item in cleaned if item["confidence"] is not None]
        current_confidence = (float(sum(confidence_values) / len(confidence_values)) if confidence_values else detector_confidence)
        if current_confidence is not None:
            current_confidence = _num(current_confidence, "detector_confidence", lo=0, hi=1)
            previous = list(self._confidence_history)
            if len(previous) >= self.surge_min_samples:
                baseline = sum(previous) / len(previous)
                self._low_confidence_streak = self._low_confidence_streak + 1 if baseline > 0 and current_confidence < baseline * self.confidence_drift_ratio else 0
                if self._low_confidence_streak >= 3:
                    event = self._emit("detector_confidence_drift_candidate", now, {"rolling_baseline": round(baseline, 4), "current_mean": round(current_confidence, 4), "low_streak": self._low_confidence_streak})
                    if event:
                        event_items.append(event)
            self._confidence_history.append(current_confidence)
        else:
            self._low_confidence_streak = 0

        return {
            "active_tracks": active_count,
            "counts_by_class": class_counts,
            "counting_lines": dict(self._counts_by_line),
            "new_crossings": len(crossing_events),
            "events": event_items,
            "quality": {
                "coordinate_system": "normalized_0_1",
                "speed_unit": "not_provided_by_this_module",
                "operator_review_required_for_alerts": True,
                "tracked_ids_in_memory": len(self._tracks),
                "history_samples": len(self._active_history),
            },
        }

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), min(200, self.event_size)))
        return list(self._events)[-limit:]
