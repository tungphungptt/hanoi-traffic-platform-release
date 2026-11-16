"""Vehicle speed in km/h from video, on a calibrated road plane.

A camera sees the road in perspective, so pixels per second say nothing about km/h. The plane
is calibrated once per camera with four points whose real distances are known (the corners of a
rectangle on the road, e.g. two lane lines and two cross-lines, measured on site or with the ruler
of a map). A homography maps each vehicle's road contact point (bottom centre of its box) to metres;
vehicles are followed with a constant-velocity tracker in those metres and each track's speed is the
slope of a straight-line fit of its position against time.

Accuracy depends on the calibration points, on how well the box bottom matches the wheels and on the
detector. Treat results from an unchecked calibration as approximate; ``draw_calibration`` overlays a
metric grid on a frame so the calibration can be judged by eye.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

MIN_TRACK_POINTS = 6
MIN_TRACK_SECONDS = 1.5
MAX_GAP_SECONDS = 1.0
MAX_SPEED_MPS = 45.0          # 162 km/h: a faster match is a wrong match
MATCH_SLACK_METERS = 2.0
MAX_MATCH_METERS = 8.0       # tighter than the speed limit alone: vehicles in a queue are 3-5 m apart
STOPPED_KMH = 5.0
MAX_PLAUSIBLE_KMH = 120.0


class CalibrationError(ValueError):
    pass


class GroundPlane:
    """Image pixels <-> metres on the road plane, from four point pairs."""

    def __init__(self, image_points: Sequence[Sequence[float]], world_points_m: Sequence[Sequence[float]]):
        image = np.asarray(image_points, dtype=np.float64)
        world = np.asarray(world_points_m, dtype=np.float64)
        if image.shape != (4, 2) or world.shape != (4, 2):
            raise CalibrationError("Cần đúng 4 điểm ảnh và 4 điểm thực (mét).")
        if not (np.isfinite(image).all() and np.isfinite(world).all()):
            raise CalibrationError("Tọa độ phải là số hữu hạn.")
        if abs(_polygon_area(image)) < 1.0 or abs(_polygon_area(world)) < 1e-6:
            raise CalibrationError("Bốn điểm không được thẳng hàng.")
        self.image_points, self.world_points = image, world
        self.to_world_matrix = cv2.getPerspectiveTransform(image.astype(np.float32), world.astype(np.float32))
        self.to_image_matrix = np.linalg.inv(self.to_world_matrix)

    @classmethod
    def from_file(cls, path: Path | str) -> "GroundPlane":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["image_points"], data["world_points_m"])

    def to_ground(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.to_world_matrix.astype(np.float64)).reshape(-1, 2)

    def to_pixels(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.to_image_matrix.astype(np.float64)).reshape(-1, 2)

    def contains_pixel(self, point: Sequence[float], margin_px: float = 0.0) -> bool:
        """Inside the calibrated quadrilateral (optionally widened): outside it the homography extrapolates."""
        quad = self.image_points.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(quad, (float(point[0]), float(point[1])), True) >= -margin_px


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _convex_in_order(points: np.ndarray) -> bool:
    """True when walking the points in the given order always turns the same way (no bow-tie, no dent)."""
    turns = []
    for i in range(len(points)):
        a, b, c = points[i], points[(i + 1) % len(points)], points[(i + 2) % len(points)]
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(t > 1e-9 for t in turns) or all(t < -1e-9 for t in turns)


def contact_point(box: Sequence[float]) -> tuple[float, float]:
    """Where the vehicle meets the road: bottom centre of its box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


@dataclass
class _Track:
    track_id: int
    times: list[float] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    class_id: int = -1

    def predict(self, t: float) -> tuple[float, float]:
        if len(self.times) < 2:
            return self.xy[-1]
        k = min(len(self.times) - 1, 4)  # velocity from the last few points
        dt = self.times[-1] - self.times[-1 - k]
        if dt <= 1e-6:
            return self.xy[-1]
        vx = (self.xy[-1][0] - self.xy[-1 - k][0]) / dt
        vy = (self.xy[-1][1] - self.xy[-1 - k][1]) / dt
        lead = t - self.times[-1]
        return self.xy[-1][0] + vx * lead, self.xy[-1][1] + vy * lead


@dataclass
class TrackSpeed:
    track_id: int
    class_id: int
    t_start: float
    t_end: float
    speed_kmh: float
    distance_m: float
    points: int


class SpeedTracker:
    """Greedy constant-velocity tracking in ground metres; ``update`` once per processed frame."""

    def __init__(self, plane: GroundPlane, *, margin_px: float = 0.0):
        self.plane = plane
        self.margin_px = margin_px
        self._next_id = 1
        self._active: dict[int, _Track] = {}
        self.finished: list[TrackSpeed] = []

    def update(self, t: float, detections: Iterable[Any]) -> None:
        points, classes = [], []
        for d in detections:
            px = contact_point(d.box)
            if self.plane.contains_pixel(px, self.margin_px):
                points.append(px)
                classes.append(d.class_id)
        ground = self.plane.to_ground(np.asarray(points)) if points else np.zeros((0, 2))
        candidates = []
        for tid, track in self._active.items():
            dt = t - track.times[-1]
            if dt <= 0 or dt > MAX_GAP_SECONDS:
                continue
            px, py = track.predict(t)
            reach = MAX_SPEED_MPS * dt + MATCH_SLACK_METERS
            for di, (gx, gy) in enumerate(ground):
                dist = math.hypot(gx - px, gy - py)
                if dist <= min(reach, MAX_MATCH_METERS):
                    candidates.append((dist, tid, di))
        candidates.sort()
        used_t, used_d = set(), set()
        for _, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            track = self._active[tid]
            track.times.append(t)
            track.xy.append((float(ground[di][0]), float(ground[di][1])))
            used_t.add(tid)
            used_d.add(di)
        for di, (gx, gy) in enumerate(ground):
            if di not in used_d:
                self._active[self._next_id] = _Track(self._next_id, [t], [(float(gx), float(gy))], classes[di])
                self._next_id += 1
        for tid in [k for k, tr in self._active.items() if t - tr.times[-1] > MAX_GAP_SECONDS]:
            self._close(self._active.pop(tid))

    def finish(self) -> list[TrackSpeed]:
        for track in self._active.values():
            self._close(track)
        self._active.clear()
        return self.finished

    def _close(self, track: _Track) -> None:
        speed = track_speed(track.times, track.xy)
        if speed is not None:
            kmh, distance = speed
            self.finished.append(TrackSpeed(track.track_id, track.class_id, track.times[0], track.times[-1], kmh, distance, len(track.times)))


def track_speed(times: Sequence[float], xy: Sequence[Sequence[float]]) -> tuple[float, float] | None:
    """Speed in km/h and path length in metres, from a least-squares line through position against time.
    None for tracks that are too short to tell a moving vehicle from detector jitter."""
    if len(times) < MIN_TRACK_POINTS or times[-1] - times[0] < MIN_TRACK_SECONDS:
        return None
    t = np.asarray(times, dtype=float)
    p = np.asarray(xy, dtype=float)
    tc = t - t.mean()
    denom = float(tc @ tc)
    if denom <= 1e-9:
        return None
    velocity = (tc[:, None] * (p - p.mean(0))).sum(0) / denom  # m/s per axis
    distance = float(np.hypot(*np.diff(p, axis=0).T).sum())
    return float(np.hypot(*velocity) * 3.6), distance


def speed_stats(speeds: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(speeds), dtype=float)
    if not arr.size:
        return {"vehicles": 0, "median_kmh": None, "mean_kmh": None, "p85_kmh": None, "stopped_share": None}
    return {"vehicles": int(arr.size), "median_kmh": round(float(np.median(arr)), 1), "mean_kmh": round(float(arr.mean()), 1),
            "p85_kmh": round(float(np.percentile(arr, 85)), 1), "stopped_share": round(float((arr < STOPPED_KMH).mean()), 3)}


def summarise(tracks: Iterable[TrackSpeed], *, window_seconds: float = 10.0, duration: float | None = None) -> list[dict[str, Any]]:
    """Per window: how many vehicles crossed, median / mean / 85th percentile speed and the share that stood still.
    A track belongs to the window of its middle time."""
    tracks = list(tracks)
    if not tracks:
        return []
    end = duration if duration is not None else max(tr.t_end for tr in tracks)
    windows: dict[int, list[TrackSpeed]] = {}
    for tr in tracks:
        windows.setdefault(int(((tr.t_start + tr.t_end) / 2.0) // window_seconds), []).append(tr)
    rows = []
    for k in range(int(end // window_seconds) + 1):
        items = windows.get(k, [])
        speeds = np.array([tr.speed_kmh for tr in items])
        rows.append({"t_start": k * window_seconds, "t_end": (k + 1) * window_seconds, "vehicles": len(items),
                     "median_kmh": round(float(np.median(speeds)), 1) if len(items) else None,
                     "mean_kmh": round(float(speeds.mean()), 1) if len(items) else None,
                     "p85_kmh": round(float(np.percentile(speeds, 85)), 1) if len(items) else None,
                     "stopped_share": round(float((speeds < STOPPED_KMH).mean()), 3) if len(items) else None})
    return rows


def draw_calibration(frame: np.ndarray, plane: GroundPlane, grid_m: float = 5.0) -> np.ndarray:
    """Frame with the calibrated quadrilateral and a metric grid, to judge a calibration by eye
    (grid lines should look parallel to the road and spaced like real distances)."""
    image = frame.copy()
    quad = plane.image_points.astype(np.int32)
    cv2.polylines(image, [quad], True, (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(quad):
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(image, f"{plane.world_points[i][0]:g},{plane.world_points[i][1]:g} m", (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    w = plane.world_points
    x0, x1, y0, y1 = w[:, 0].min(), w[:, 0].max(), w[:, 1].min(), w[:, 1].max()
    for gx in np.arange(x0, x1 + 1e-6, grid_m):
        a, b = plane.to_pixels(np.array([[gx, y0], [gx, y1]]))
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 0), 1, cv2.LINE_AA)
    for gy in np.arange(y0, y1 + 1e-6, grid_m):
        a, b = plane.to_pixels(np.array([[x0, gy], [x1, gy]]))
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 0), 1, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------- from a web request and over a whole video

MAX_SIDE_METERS = 300.0
MIN_QUAD_SHARE = 0.005   # the calibrated quadrilateral must cover at least 0.5% of the frame


def plane_from_request(data: Any, frame_width: int, frame_height: int) -> tuple[GroundPlane, dict[str, float]]:
    """Calibration sent by the web page: four clicked points as fractions of the frame (near-left, near-right,
    far-right, far-left of a rectangle on the road) and the real width and length of that rectangle in metres."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError as exc:
            raise CalibrationError("Calibration không phải JSON hợp lệ.") from exc
    if not isinstance(data, dict):
        raise CalibrationError("Calibration phải là một đối tượng.")
    points = data.get("points")
    if (not isinstance(points, list) or len(points) != 4
            or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points)):
        raise CalibrationError("Cần đúng 4 điểm [x, y].")
    try:
        norm = np.asarray(points, dtype=float)
        width_m, length_m = float(data.get("width_m")), float(data.get("length_m"))
        when = float(data.get("time_seconds") or 0.0)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("Điểm, bề rộng và chiều dài phải là số.") from exc
    if not (np.isfinite(norm).all() and math.isfinite(width_m) and math.isfinite(length_m) and math.isfinite(when)):
        raise CalibrationError("Giá trị phải là số hữu hạn.")
    if norm.min() < 0.0 or norm.max() > 1.0:
        raise CalibrationError("Điểm phải nằm trong khung hình.")
    if not (1.0 <= width_m <= MAX_SIDE_METERS and 1.0 <= length_m <= MAX_SIDE_METERS):
        raise CalibrationError(f"Bề rộng và chiều dài phải trong khoảng 1–{int(MAX_SIDE_METERS)} m.")
    pixels = norm * np.array([frame_width, frame_height], dtype=float)
    area = abs(_polygon_area(pixels))
    if not _convex_in_order(pixels) or area < MIN_QUAD_SHARE * frame_width * frame_height:
        raise CalibrationError("Bốn điểm phải tạo thành một tứ giác lồi, không quá nhỏ (không tự cắt nhau).")
    plane = GroundPlane(pixels, [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])
    return plane, {"width_m": width_m, "length_m": length_m, "time_seconds": max(0.0, when)}


def measure_bursts(video: Path, detector: Any, plane: GroundPlane, *, intervals: Sequence[Sequence[float]] | None = None,
                   fps: float = 4.0, burst_seconds: float = 12.0, every_seconds: float = 120.0,
                   max_frames: int = 1500) -> dict[str, Any]:
    """Speed over short bursts spread across the video (a speed needs a vehicle followed for a few seconds,
    not every second of an hour). Only inside ``intervals`` (when the calibrated camera is on air)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise CalibrationError("Không mở được video để đo tốc độ.")
    native = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / native
    stride = max(1, int(round(native / fps)))
    allowed = [list(map(float, i)) for i in intervals] if intervals else [[0.0, total]]
    starts = []
    t = 0.0
    while t < total:
        for a, b in allowed:
            if a <= t and t + burst_seconds <= b + 1e-6:
                starts.append(t)
                break
        t += every_seconds
    if not starts:  # a short or fragmented video: take the first long enough on-air stretch
        for a, b in allowed:
            if b - a >= MIN_TRACK_SECONDS * 2:
                starts.append(a)
                break
    per_burst = int(burst_seconds * fps) + 1
    if len(starts) * per_burst > max_frames:
        keep = max(1, max_frames // max(1, per_burst))
        starts = [starts[int(i * len(starts) / keep)] for i in range(keep)]
    bursts, all_speeds, frames = [], [], 0
    try:
        for start in starts:
            end = min(start + burst_seconds, total)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * native))
            tracker = SpeedTracker(plane)
            index = int(start * native)
            while index / native < end:
                if (index - int(start * native)) % stride:
                    if not cap.grab():
                        break
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    tracker.update(index / native, detector.detect(frame))
                    frames += 1
                index += 1
            speeds = [tr.speed_kmh for tr in tracker.finish()]
            all_speeds += speeds
            bursts.append({"t_start": round(start, 1), "t_end": round(end, 1), **speed_stats(speeds)})
    finally:
        cap.release()
    overall = speed_stats(all_speeds)
    warning = None
    if overall["median_kmh"] is not None and overall["median_kmh"] > MAX_PLAUSIBLE_KMH:
        warning = "Tốc độ trung vị vượt 120 km/h: gần như chắc chắn khoảng cách hoặc bốn điểm đã nhập sai."
    return {"bursts": bursts, "overall": overall, "frames_processed": frames, "burst_seconds": burst_seconds,
            "warning": warning}
