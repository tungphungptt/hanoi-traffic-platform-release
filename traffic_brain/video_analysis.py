"""Turn a traffic video into a visual congestion series and +5/+10/+15 minute forecasts.

Per sampled frame (1 fps by default): vehicles are detected, tracked between samples, and an
explainable congestion index (CI, 0..1) is computed from box occupancy, vehicle count and
the share of stopped motion. The CI series then feeds ``minute_forecaster``.

Broadcast recordings (for example the Hanoi traffic-camera programme) cut between several
cameras and studio shots. Every sample therefore also gets a scene label from a small image
signature; each camera gets its own series with gaps while it is off air, and forecasts are
made per camera. A video from one fixed camera ends up as a single scene.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

from traffic_brain.detector import (CONFIDENCE_THRESHOLD, PERSON_CLASS_ID, VEHICLE_CLASS_IDS, Detector,
                                    update_tracks)
from traffic_brain.minute_forecaster import build_minute_forecast, build_observed_series
from traffic_brain.traffic_levels import BUSY_CI, SEVERE_CI, level_for_ci

# Scene segmentation (fixed before evaluation; see tests/test_video_analysis.py).
SIGNATURE_SIZE = (32, 18)
SCENE_SIMILARITY = 0.80        # join an existing scene when the signature correlation is at least this
SCENE_MERGE_SIMILARITY = 0.70  # merge scenes whose mean signatures are this similar (same camera, light change)
MIN_RUN_SECONDS = 5.0          # shorter runs are cuts/transitions, not camera time
MIN_CAMERA_SECONDS = 60.0      # a camera needs at least this much air time to be reported
SINGLE_CAMERA_SHARE = 0.90     # one scene covering this share of usable time = fixed-camera video
BANNER_RED_SHARE = 0.08        # red overlay (programme banner, studio) in the top or bottom band


class VideoAnalysisError(ValueError):
    """A problem with the input video that the user can fix."""


def visual_ci(vehicle_count: int, occupancy: float, normalized_speed: float | None) -> float:
    """Explainable congestion index: occupancy, count and (when tracked) stopped motion."""
    occ = min(max(float(occupancy), 0.0) / 0.22, 1.0)
    count = min(max(int(vehicle_count), 0) / 20.0, 1.0)
    if normalized_speed is None:
        ci = 0.62 * occ + 0.38 * count
    else:
        stopped = 1.0 - min(max(float(normalized_speed), 0.0) / 0.045, 1.0)
        ci = 0.50 * occ + 0.32 * count + 0.18 * stopped
    return max(0.0, min(1.0, ci))


def frame_signature(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """(normalised 32x18 grey thumbnail, has a programme banner/studio overlay)."""
    small = cv2.resize(frame, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    grey = (grey - grey.mean()) / (grey.std() + 1e-6)
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    band = int(90 * 0.3)
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 80)
    banner = bool(red[:band].mean() > BANNER_RED_SHARE or red[-band:].mean() > BANNER_RED_SHARE)
    return grey, banner


def segment_scenes(signatures: list[np.ndarray | None], times: list[float]) -> list[int]:
    """Label each sample with a camera id (0, 1, ... by air time) or -1 (banner, studio, cut)."""
    n = len(signatures)
    labels = np.full(n, -1, dtype=int)
    cents: list[np.ndarray] = []
    counts: list[int] = []
    for i, sig in enumerate(signatures):
        if sig is None:
            continue
        if cents:
            sims = np.array([float(np.dot(sig, c) / sig.size) for c in cents])
            k = int(sims.argmax())
            if sims[k] >= SCENE_SIMILARITY:
                cents[k] = (cents[k] * counts[k] + sig) / (counts[k] + 1)
                counts[k] += 1
                labels[i] = k
                continue
        cents.append(sig.astype(np.float64).copy())
        counts.append(1)
        labels[i] = len(cents) - 1
    # Merge scenes that are the same camera (mean signatures still alike).
    parent = list(range(len(cents)))

    def root(k: int) -> int:
        while parent[k] != k:
            k = parent[k]
        return k
    order = sorted(range(len(cents)), key=lambda k: -counts[k])
    for a_i, a in enumerate(order):
        for b in order[a_i + 1:]:
            ra, rb = root(a), root(b)
            if ra != rb and float(np.dot(cents[a], cents[b]) / cents[a].size) >= SCENE_MERGE_SIMILARITY:
                parent[rb] = ra
    labels = np.array([root(k) if k >= 0 else -1 for k in labels])
    # Short runs are transitions.
    period = float(np.median(np.diff(times))) if n > 1 else 1.0
    min_run = max(1, int(round(MIN_RUN_SECONDS / max(period, 1e-6))))
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if j - i < min_run:
            labels[i:j] = -1
        i = j
    # Keep cameras with enough air time, renumbered by air time.
    ids, sizes = np.unique(labels[labels >= 0], return_counts=True)
    keep = [int(k) for k, s in sorted(zip(ids, sizes), key=lambda x: -x[1]) if s * period >= MIN_CAMERA_SECONDS]
    remap = {k: new for new, k in enumerate(keep)}
    return [remap.get(int(k), -1) for k in labels]


def _intervals(times: list[float], mask: list[bool], period: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t, on in zip(times, mask):
        if not on:
            continue
        if out and t - out[-1][1] <= period * 1.5 + 1e-6:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 1), round(b + period, 1)] for a, b in out]


def _camera_series(timeline: list[dict[str, Any]], camera: int | None) -> dict[str, np.ndarray]:
    rows = timeline if camera is None else [r for r in timeline if r["scene"] == camera]
    return build_observed_series(rows)


class FrameMeter:
    """Per-frame measurement shared by the video analysis and the live collector.

    Call ``measure(frame, t)`` on samples taken at ``sample_rate`` per second. Vehicles are tracked
    between calls; a camera cut (low signature correlation with the previous sample) resets tracking.
    """

    def __init__(self, detector: Detector, sample_rate: float):
        self.detector = detector
        self.sample_rate = float(sample_rate)
        self.tracks: dict = {}
        self.next_id = 1
        self.prev_sig: np.ndarray | None = None
        self.sample = 0

    def measure(self, frame: np.ndarray, t: float) -> tuple[dict[str, Any], np.ndarray, bool]:
        """Returns (timeline row, signature, banner flag)."""
        sig, banner = frame_signature(frame)
        # A camera cut: vehicles of the previous shot are not the same vehicles.
        cut = self.prev_sig is None or float(np.dot(sig, self.prev_sig) / sig.size) < SCENE_SIMILARITY
        self.prev_sig = sig
        if cut:
            self.tracks = {}
        detections = self.detector.detect(frame)
        self.tracks, self.next_id, moves = update_tracks(detections, self.tracks, self.next_id, self.sample)
        speeds = [d * self.sample_rate / e for d, e in moves if e > 0]
        diagonal = math.hypot(frame.shape[1], frame.shape[0]) or 1.0
        normalized_speed = float(np.median(speeds)) / diagonal if len(speeds) >= 2 else None
        area = max(1, frame.shape[0] * frame.shape[1])
        occupancy = min(1.0, sum(max(0.0, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])) for d in detections) / area)
        self.sample += 1
        row = {
            "timestamp_seconds": round(t, 3),
            "raw_ci": visual_ci(len(detections), occupancy, normalized_speed),
            "vehicle_count": len(detections),
            "occupancy_ratio": round(occupancy, 5),
            "normalized_speed": None if normalized_speed is None else round(normalized_speed, 5),
            # Share of vehicles that were not near a vehicle of the previous sample: high when
            # traffic moves through the frame, low when the same vehicles stand still.
            "turnover": round((len(detections) - len(moves)) / len(detections), 4) if detections and not cut else None,
        }
        return row, sig, banner


def analyze_video(video_path: Path, detector: Detector, *, max_duration_seconds: float,
                  timing_confirmed: bool = False, sample_fps: float = 1.0,
                  library: Iterable[Mapping[str, np.ndarray]] = (), library_videos: int = 0,
                  progress: Callable[[int, int, int], None] | None = None) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoAnalysisError("Không mở được video; kiểm tra codec/định dạng.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if fps > 0 and total > 0 else 0.0
    if not math.isfinite(fps) or fps <= 0 or duration <= 0:
        cap.release()
        raise VideoAnalysisError("Video cần FPS, số frame và thời lượng hợp lệ.")
    if duration > max_duration_seconds:
        cap.release()
        raise VideoAnalysisError(f"Video dài hơn giới hạn {int(max_duration_seconds // 60)} phút.")
    stride = max(1, int(round(fps / max(0.25, min(2.0, sample_fps)))))
    sample_rate = fps / stride
    timeline: list[dict[str, Any]] = []
    signatures: list[np.ndarray | None] = []
    meter = FrameMeter(detector, sample_rate)
    index = 0
    last_pct = -1
    try:
        while True:
            if index % stride:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                row, sig, banner = meter.measure(frame, index / fps)
                signatures.append(None if banner else sig)
                timeline.append(row)
            index += 1
            pct = min(99, index * 100 // max(1, total))
            if progress and pct != last_pct:
                last_pct = pct
                progress(pct, index, total)
    finally:
        cap.release()
    if len(timeline) < 2:
        raise VideoAnalysisError("Không có đủ khung hình để phân tích.")

    times = [r["timestamp_seconds"] for r in timeline]
    labels = segment_scenes(signatures, times)
    for row, label in zip(timeline, labels):
        row["scene"] = label
    usable = sum(1 for k in labels if k >= 0)
    main_share = (sum(1 for k in labels if k == 0) / usable) if usable else 0.0
    single = usable == 0 or main_share >= SINGLE_CAMERA_SHARE
    period = 1.0 / sample_rate
    library = list(library)
    cameras = []
    for cam in ([None] if single else sorted({k for k in labels if k >= 0})):
        mask = [True] * len(timeline) if cam is None else [k == cam for k in labels]
        series = _camera_series(timeline, cam)
        forecast = build_minute_forecast((), duration_seconds=duration, timing_confirmed=timing_confirmed,
                                         library=library, library_videos=library_videos, series=series)
        cameras.append({
            "id": 0 if cam is None else cam,
            "label": "Camera" if cam is None else f"Camera {cam + 1}",
            "on_air_seconds": round(sum(mask) * period, 1),
            "intervals": _intervals(times, mask, period),
            "series": series,
            "minute_forecast": forecast,
        })
    for cam in cameras:
        anchors = cam["minute_forecast"].get("anchors") or []
        cam["checked_forecasts"] = sum(1 for a in anchors for h in a["horizons"] if h.get("actual_ci") is not None)
    return {
        "video": {"duration_seconds": round(duration, 3), "fps": round(fps, 3), "width": width,
                  "height": height, "frames_total": total, "sample_fps": round(sample_rate, 3)},
        "mode": "single_camera" if single else "broadcast",
        "cameras": cameras,
        "timeline": timeline,
    }


def result_payload(analysis: Mapping[str, Any], filename: str) -> dict[str, Any]:
    """What the web page receives: per-camera forecasts, the best-checked camera first in view."""
    cameras = [{k: v for k, v in cam.items() if k != "series"} for cam in analysis["cameras"]]
    main = max(cameras, key=lambda c: (c["checked_forecasts"], c["on_air_seconds"]))
    return {"filename": filename, "video": analysis["video"], "mode": analysis["mode"],
            "cameras": cameras, "main_camera": main["id"], "minute_forecast": main["minute_forecast"]}


# ---------------------------------------------------------------- evidence frames

EVIDENCE_WIDTH = 640
EVIDENCE_JPEG_QUALITY = 72
_LEVEL_BGR = {"free": (80, 160, 30), "busy": (30, 150, 230), "gridlock": (40, 40, 200)}
EVIDENCE_REASONS = {
    "up_busy": "Bắt đầu đông đúc",
    "up_gridlock": "Bắt đầu tắc nghẽn",
    "down_busy": "Hết tắc, còn đông",
    "down_free": "Thông thoáng trở lại",
    "peak": "Đông nhất",
    "low": "Thông thoáng nhất",
    "even": "Mốc định kỳ",
}


def pick_evidence(cameras: Iterable[Mapping[str, Any]], *, limit: int = 24, per_camera: int = 8,
                  min_gap_seconds: float = 60.0) -> list[dict[str, Any]]:
    """Moments worth a picture: level changes, the busiest and calmest points, then even spacing."""
    cameras = list(cameras)
    if not cameras:
        return []
    budget = max(3, min(per_camera, limit // len(cameras)))
    picks: list[dict[str, Any]] = []
    for cam in cameras:
        t = np.asarray(cam["series"]["t"], dtype=float)
        ci = np.asarray(cam["series"]["ci"], dtype=float)
        ok = np.isfinite(ci)
        if ok.sum() < 2:
            continue
        chosen: list[tuple[float, str, float]] = []

        def add(i: int, reason: str) -> None:
            if all(abs(t[i] - other[0]) >= min_gap_seconds for other in chosen):
                chosen.append((float(t[i]), reason, float(ci[i])))

        idx = np.flatnonzero(ok)
        add(int(idx[int(np.argmax(ci[idx]))]), "peak")
        for prev, cur in zip(idx[:-1], idx[1:]):
            if t[cur] - t[prev] > 90:  # off air in between: not a real change
                continue
            a, b = ci[prev], ci[cur]
            if a < SEVERE_CI <= b:
                add(int(cur), "up_gridlock")
            elif a < BUSY_CI <= b < SEVERE_CI:
                add(int(cur), "up_busy")
            elif b < SEVERE_CI <= a:
                add(int(cur), "down_busy")
            elif b < BUSY_CI <= a:
                add(int(cur), "down_free")
        add(int(idx[int(np.argmin(ci[idx]))]), "low")
        if len(chosen) < budget:
            for q in np.linspace(0.1, 0.9, budget):
                if len(chosen) >= budget:
                    break
                add(int(idx[int(q * (len(idx) - 1))]), "even")
        # Gridlock first when trimming, then level changes, then the rest.
        order = {"up_gridlock": 0, "peak": 1, "up_busy": 2, "down_busy": 3, "down_free": 4, "low": 5, "even": 6}
        chosen = sorted(sorted(chosen, key=lambda c: order[c[1]])[:budget], key=lambda c: c[0])
        for when, reason, value in chosen:
            picks.append({"camera": cam["id"], "t": round(when, 3), "ci": round(value, 4),
                          "level_key": level_for_ci(value)[0], "reason": reason,
                          "reason_label": EVIDENCE_REASONS[reason]})
    picks.sort(key=lambda p: (p["camera"], p["t"]))
    return picks[:limit] if len(picks) <= limit else sorted(
        sorted(picks, key=lambda p: ("up_gridlock", "peak", "up_busy", "down_busy", "down_free", "low", "even").index(p["reason"]))[:limit],
        key=lambda p: (p["camera"], p["t"]))


PRIVACY_MIN_SCORE = 0.12  # lower than the counting threshold: a missed face is worse than an extra blur
PRIVACY_PADDING = 0.15


def blur_private(frame: np.ndarray, detections: Iterable) -> np.ndarray:
    """Copy of ``frame`` with every detected person and vehicle pixelated beyond recognition.

    Faces (riders, pedestrians) and number plates sit inside those boxes, so the whole box is
    pixelated into a few coarse cells, which no zoom can undo. The original pixels are never kept.
    """
    out = frame.copy()
    height, width = out.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d.box
        pad_x, pad_y = (x2 - x1) * PRIVACY_PADDING, (y2 - y1) * PRIVACY_PADDING
        x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        x2, y2 = min(width, int(x2 + pad_x) + 1), min(height, int(y2 + pad_y) + 1)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = out[y1:y2, x1:x2]
        cells_x, cells_y = max(2, (x2 - x1) // 14), max(2, (y2 - y1) // 14)
        small = cv2.resize(region, (cells_x, cells_y), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def draw_evidence(frame: np.ndarray, detections: list, pick: Mapping[str, Any]) -> np.ndarray:
    """Frame with the detected vehicles boxed and a short ASCII caption (OpenCV fonts lack Vietnamese)."""
    height, width = frame.shape[:2]
    scale = EVIDENCE_WIDTH / float(width) if width > EVIDENCE_WIDTH else 1.0
    image = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
    colour = _LEVEL_BGR[pick["level_key"]]
    for d in detections:
        x1, y1, x2, y2 = (int(v * scale) for v in d.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    minutes, seconds = divmod(int(pick["t"]), 60)
    hours, minutes = divmod(minutes, 60)
    text = f"{hours:d}:{minutes:02d}:{seconds:02d}  CI {pick['ci']:.2f}  {len(detections)} xe"
    cv2.rectangle(image, (0, 0), (image.shape[1], 24), (20, 20, 20), -1)
    cv2.putText(image, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (6, 24), colour, -1)
    return image


def capture_evidence(video_path: Path, detector: Detector, cameras: Iterable[Mapping[str, Any]],
                     out_dir: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    """Save boxed key frames under ``out_dir`` (``NN.jpg``); returns the picks with their file name."""
    picks = pick_evidence(cameras, limit=limit)
    if not picks:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved: list[dict[str, Any]] = []
    try:
        for pick in picks:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(pick["t"]) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            found = detector.detect(frame, VEHICLE_CLASS_IDS | {PERSON_CLASS_ID}, PRIVACY_MIN_SCORE)
            counted = [d for d in found if d.class_id in VEHICLE_CLASS_IDS and d.score >= CONFIDENCE_THRESHOLD]
            image = draw_evidence(blur_private(frame, found), counted, pick)
            name = f"{len(saved) + 1:02d}.jpg"
            if cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY]):
                saved.append({**pick, "file": name})
    finally:
        cap.release()
    return saved
