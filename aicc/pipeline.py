"""AICC-compatible video inference, integrated as an optional local pipeline.

Model artifacts are expected in models/ and are not bundled with this source ZIP.
The predictor labels are the two classes declared by models/manifest.json.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from aicc.core.preprocessing import preprocess_frame
from traffic_brain.analytics import adaptive_frame_stride

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = MODEL_DIR / "manifest.json"
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
WINDOW_FRAMES = 30  # legacy documentation reference; runtime windows are one second of sampled frames
WINDOW_SECONDS = 1.0
DEFAULT_SAMPLE_FPS = 6.0
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
TRACK_DISTANCE_THRESHOLD = 50.0
MAX_TRACK_AGE = 5
COCO_CLASS_IDS = {"car": 2, "truck": 7, "bus": 5}
MAX_VIDEO_BYTES = 500 * 1024 * 1024


@dataclass
class Detection:
    class_id: int
    box: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    track_id: int
    class_id: int
    center: tuple[float, float]
    last_frame: int


class AICCRuntime:
    """Loads the AICC ONNX/scaler bundle on demand; missing weights don't break dashboard startup."""
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundle: dict[str, Any] | None = None
        self._loaded_signature: tuple[tuple[str, int, int], ...] | None = None
        self._error: str | None = None

    def _read_manifest(self) -> dict[str, Any]:
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Thiếu manifest AICC: {MANIFEST_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manifest AICC không phải JSON hợp lệ: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Manifest AICC phải là JSON object")
        if not isinstance(payload.get("version"), str) or not payload["version"]:
            raise ValueError("Manifest AICC thiếu version")
        if not isinstance(payload.get("sequence_length"), int) or payload["sequence_length"] < 1:
            raise ValueError("Manifest AICC cần sequence_length là số nguyên dương")
        if not isinstance(payload.get("classes"), list) or not payload["classes"]:
            raise ValueError("Manifest AICC cần classes không rỗng")
        models = payload.get("models")
        if not isinstance(models, dict):
            raise ValueError("Manifest AICC cần models object")
        for key in ("detector", "predictor", "scaler"):
            value = models.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Manifest AICC thiếu models.{key}")
        return payload

    def _paths_and_signature(self) -> tuple[dict[str, Path], tuple[tuple[str, int, int], ...], dict[str, Any]]:
        manifest = self._read_manifest()
        paths: dict[str, Path] = {}
        signature_items: list[tuple[str, int, int]] = []
        for key, value in manifest["models"].items():
            candidate = Path(value)
            path = candidate if candidate.is_absolute() else ROOT / candidate
            path = path.resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError(f"Đường dẫn artifact không hợp lệ: {key}")
            paths[key] = path
            if path.is_file():
                st = path.stat()
                signature_items.append((key, st.st_mtime_ns, st.st_size))
            else:
                signature_items.append((key, -1, -1))
        return paths, tuple(sorted(signature_items)), manifest

    def status(self) -> dict[str, Any]:
        try:
            paths, signature, manifest = self._paths_and_signature()
            missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
            loaded = self._bundle is not None and signature == self._loaded_signature
            return {
                "ready": not missing and loaded,
                "artifacts_present": not missing,
                "loaded": loaded,
                "model_version": manifest.get("version"),
                "classes": manifest.get("classes", []),
                "sequence_length": manifest.get("sequence_length"),
                "features": manifest.get("features", []),
                "files": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                "missing_files": missing,
                "error": self._error,
                "provider": "CPUExecutionProvider",
            }
        except Exception as exc:
            return {"ready": False, "artifacts_present": False, "loaded": False, "error": str(exc), "missing_files": []}

    def load(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            paths, signature, manifest = self._paths_and_signature()
            missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
            if missing:
                self._bundle = None
                self._loaded_signature = None
                self._error = "Thiếu model artifacts: " + ", ".join(missing)
                raise FileNotFoundError(self._error)
            if not force and self._bundle is not None and signature == self._loaded_signature:
                return self._bundle
            try:
                import onnxruntime as ort
                from sklearn.preprocessing import MinMaxScaler
                detector = ort.InferenceSession(str(paths["detector"]), providers=["CPUExecutionProvider"])
                predictor = ort.InferenceSession(str(paths["predictor"]), providers=["CPUExecutionProvider"])
                # Only unpickle a scaler artifact from a trusted source: pickle can execute code.
                with paths["scaler"].open("rb") as stream:
                    scaler = pickle.load(stream)
                if not isinstance(scaler, MinMaxScaler):
                    raise TypeError("scaler.pkl không phải fitted sklearn MinMaxScaler")
                vehicle_classes = manifest.get("vehicle_classes", ["car", "truck", "bus"])
                unsupported = [name for name in vehicle_classes if name not in COCO_CLASS_IDS]
                if unsupported:
                    raise ValueError("Chưa cấu hình class id cho: " + ", ".join(unsupported))
                self._bundle = {
                    "manifest": manifest,
                    "detector": detector,
                    "predictor": predictor,
                    "scaler": scaler,
                    "detector_input_name": detector.get_inputs()[0].name,
                    "predictor_input_name": predictor.get_inputs()[0].name,
                    "predictor_output_name": predictor.get_outputs()[0].name,
                    "class_ids": {COCO_CLASS_IDS[name] for name in vehicle_classes},
                }
                self._loaded_signature = signature
                self._error = None
                return self._bundle
            except Exception as exc:
                self._bundle = None
                self._loaded_signature = None
                self._error = f"Không nạp được AICC model bundle: {exc}"
                raise RuntimeError(self._error) from exc


def _detect(detector: Any, input_name: str, frame: np.ndarray, original_shape: tuple[int, int], class_ids: set[int]) -> list[Detection]:
    input_tensor = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
    output = np.asarray(detector.run(None, {input_name: input_tensor})[0]).squeeze()
    if output.ndim != 2:
        raise RuntimeError(f"Unexpected detector output shape: {output.shape}")
    if output.shape[0] < output.shape[1] and output.shape[0] != 6:
        output = output.T
    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    classes: list[int] = []
    if output.shape[1] == 6:
        for row in output:
            class_id, score = int(row[5]), float(row[4])
            if class_id in class_ids and score >= CONFIDENCE_THRESHOLD:
                boxes.append(tuple(map(float, row[:4])))
                scores.append(score)
                classes.append(class_id)
    elif output.shape[1] > 6:
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if class_id not in class_ids or score < CONFIDENCE_THRESHOLD:
                continue
            cx, cy, width, height = map(float, row[:4])
            boxes.append((cx-width/2, cy-height/2, cx+width/2, cy+height/2))
            scores.append(score)
            classes.append(class_id)
    else:
        raise RuntimeError(f"Unexpected detector output shape: {output.shape}")
    kept: list[int] = []
    for class_id in class_ids:
        indices = [i for i, found in enumerate(classes) if found == class_id]
        if not indices:
            continue
        boxes_xywh = [[boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]] for i in indices]
        nms = cv2.dnn.NMSBoxes(boxes_xywh, [scores[i] for i in indices], CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        kept.extend(indices[int(i)] for i in np.asarray(nms).reshape(-1))
    oh, ow = original_shape
    sx, sy = ow / frame.shape[1], oh / frame.shape[0]
    return [Detection(classes[i], (boxes[i][0]*sx, boxes[i][1]*sy, boxes[i][2]*sx, boxes[i][3]*sy), float(scores[i])) for i in kept]


def _update_tracks(detections: list[Detection], tracks: dict[int, Track], next_track_id: int, frame_index: int) -> tuple[dict[int, Track], int, list[tuple[int, Detection, float, int]]]:
    candidates: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        for track_id, track in tracks.items():
            if track.class_id != det.class_id or frame_index - track.last_frame > MAX_TRACK_AGE:
                continue
            distance = float(np.hypot(det.center[0]-track.center[0], det.center[1]-track.center[1]))
            if distance <= TRACK_DISTANCE_THRESHOLD:
                candidates.append((distance, di, track_id))
    candidates.sort()
    assigned_dets: set[int] = set()
    assigned_tracks: set[int] = set()
    associations: list[tuple[int, Detection, float, int]] = []
    updated: dict[int, Track] = {}
    for distance, di, track_id in candidates:
        if di in assigned_dets or track_id in assigned_tracks:
            continue
        det = detections[di]
        prev = tracks[track_id]
        elapsed = frame_index - prev.last_frame
        updated[track_id] = Track(track_id, det.class_id, det.center, frame_index)
        associations.append((track_id, det, distance, elapsed))
        assigned_dets.add(di)
        assigned_tracks.add(track_id)
    for track_id, track in tracks.items():
        if track_id not in assigned_tracks and frame_index-track.last_frame <= MAX_TRACK_AGE:
            updated[track_id] = track
    for di, det in enumerate(detections):
        if di not in assigned_dets:
            track_id = next_track_id
            next_track_id += 1
            updated[track_id] = Track(track_id, det.class_id, det.center, frame_index)
            associations.append((track_id, det, 0.0, 0))
    return updated, next_track_id, associations


def _predict_window_with_confidence(bundle: dict[str, Any], sequence: deque[np.ndarray]) -> tuple[int, float, np.ndarray]:
    raw = np.stack(sequence).astype(np.float32)
    scaled = bundle["scaler"].transform(raw).astype(np.float32)
    tensor = np.ascontiguousarray(scaled[np.newaxis, ...])
    logits = bundle["predictor"].run([bundle["predictor_output_name"]], {bundle["predictor_input_name"]: tensor})[0]
    logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    shifted = logits_arr - np.max(logits_arr)
    exp_vals = np.exp(shifted)
    denom = float(np.sum(exp_vals))
    probs = exp_vals / (denom if denom > 1e-9 else 1.0)
    label_id = int(np.argmax(probs))
    confidence = float(probs[label_id])
    return label_id, confidence, probs


def _predict_window(bundle: dict[str, Any], sequence: deque[np.ndarray]) -> int:
    label_id, _, _ = _predict_window_with_confidence(bundle, sequence)
    return label_id


class AICCSequenceStabilizer:
    """Strictly causal online smoother for AICC 2-class sequence predictions.

    Uses causal probability smoothing (EMA) combined with hysteresis around the
    decision boundary and minimum state duration to eliminate rapid 1-second label toggling.
    """
    def __init__(
        self,
        hysteresis_margin: float = 0.08,
        min_consecutive_windows: int = 2,
        ema_alpha: float = 0.40,
    ) -> None:
        self.hysteresis = float(hysteresis_margin)
        self.min_consecutive = int(min_consecutive_windows)
        self.ema_alpha = float(ema_alpha)
        self.smoothed_prob_busy: float | None = None
        self.current_label_id: int | None = None
        self.candidate_label_id: int | None = None
        self.candidate_count: int = 0

    def update(self, raw_label_id: int, probs: np.ndarray) -> tuple[int, float]:
        """Update stabilizer with a new observation (strictly causal)."""
        prob_busy = float(probs[1]) if len(probs) > 1 else (1.0 if raw_label_id == 1 else 0.0)
        if self.smoothed_prob_busy is None:
            self.smoothed_prob_busy = prob_busy
            self.current_label_id = raw_label_id
            self.candidate_label_id = raw_label_id
            self.candidate_count = 1
            return self.current_label_id, self.smoothed_prob_busy

        self.smoothed_prob_busy = self.ema_alpha * prob_busy + (1.0 - self.ema_alpha) * self.smoothed_prob_busy
        self.smoothed_prob_busy = max(0.0, min(1.0, self.smoothed_prob_busy))

        if self.current_label_id == 0:
            candidate = 1 if self.smoothed_prob_busy >= (0.50 + self.hysteresis) else 0
        else:
            candidate = 0 if self.smoothed_prob_busy <= (0.50 - self.hysteresis) else 1

        if candidate == self.current_label_id:
            self.candidate_label_id = candidate
            self.candidate_count = 0
        else:
            if candidate == self.candidate_label_id:
                self.candidate_count += 1
            else:
                self.candidate_label_id = candidate
                self.candidate_count = 1
            if self.candidate_count >= self.min_consecutive:
                self.current_label_id = candidate
                self.candidate_count = 0

        return self.current_label_id, self.smoothed_prob_busy


def process_video(
    video_path: Path,
    bundle: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    sample_fps_target: float | None = None,
) -> dict[str, Any]:
    """Run the upstream AICC contract on sampled frames for more responsive CPU inference.

    The detector is called at a bounded sampling rate (default 6 FPS). Video is still
    decoded sequentially, while tracking is measured in sampled-frame time so the
    pixel/second feature remains correctly scaled. The model contract remains exactly
    [car_count, avg_speed_pixels_per_second, speed_cv] over a sequence of short windows.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Không mở được video; kiểm tra codec/định dạng")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video không có metadata FPS hợp lệ")
    target_fps = sample_fps_target
    if target_fps is None:
        try: target_fps = float(os.getenv("AICC_SAMPLE_FPS", str(DEFAULT_SAMPLE_FPS)))
        except ValueError: target_fps = DEFAULT_SAMPLE_FPS
    if not math.isfinite(target_fps) or target_fps <= 0:
        target_fps = DEFAULT_SAMPLE_FPS
    target_fps = max(1.0, min(12.0, target_fps))
    # Shared, tested CPU frame-budget policy. The detector remains bounded by 12 FPS.
    sampling = adaptive_frame_stride(fps, target_fps, max_target_fps=12.0)
    stride = int(sampling["frame_stride"])
    sample_fps = float(sampling["sample_fps"])
    window_samples = max(1, int(round(sample_fps * WINDOW_SECONDS)))

    sequence_length = int(bundle["manifest"]["sequence_length"])
    feature_sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
    results: list[dict[str, Any]] = []
    tracks: dict[int, Track] = {}
    next_id = source_frame_index = sample_index = window_start_sample = 0
    present_track_ids: set[int] = set()
    track_motion: dict[int, list[float]] = {}
    class_ids = bundle["class_ids"]
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    last_progress = -1
    stabilizer = AICCSequenceStabilizer()

    def finish_window(window_sample_count: int, end_source_frame: int) -> None:
        nonlocal present_track_ids, track_motion, window_start_sample
        speeds = [m[0] / (m[1] / sample_fps) for m in track_motion.values() if m[1] > 0]
        average_speed = float(np.mean(speeds)) if speeds else 0.0
        speed_cv = float(np.std(speeds) / average_speed) if speeds and average_speed > 0 else 0.0
        features = np.asarray([len(present_track_ids), average_speed, speed_cv], dtype=np.float32)
        feature_sequence.append(features)
        if len(feature_sequence) == sequence_length:
            raw_label_id, confidence, probs = _predict_window_with_confidence(bundle, feature_sequence)
            smoothed_label_id, smoothed_prob = stabilizer.update(raw_label_id, probs)
            classes = bundle["manifest"].get("classes", [])
            raw_label = classes[raw_label_id] if 0 <= raw_label_id < len(classes) else f"class_{raw_label_id}"
            raw_translated = {"Thong_thoang": "Thông thoáng", "Dong_duc": "Đông đúc"}.get(raw_label, raw_label)
            smoothed_label = classes[smoothed_label_id] if 0 <= smoothed_label_id < len(classes) else f"class_{smoothed_label_id}"
            smoothed_translated = {"Thong_thoang": "Thông thoáng", "Dong_duc": "Đông đúc"}.get(smoothed_label, smoothed_label)
            results.append({
                "timestamp_seconds": round(end_source_frame / fps, 3),
                "class_id": smoothed_label_id,
                "congestion_class": smoothed_label,
                "congestion_label": smoothed_translated,
                "raw_class_id": raw_label_id,
                "raw_congestion_class": raw_label,
                "raw_congestion_label": raw_translated,
                "smoothed_class_id": smoothed_label_id,
                "smoothed_congestion_class": smoothed_label,
                "smoothed_congestion_label": smoothed_translated,
                "model_confidence": round(confidence, 4),
                "smoothed_probability_busy": round(smoothed_prob, 4),
                "vehicle_tracks_in_window": len(present_track_ids),
                "avg_speed_pixels_per_second": round(average_speed, 3),
                "speed_cv": round(speed_cv, 4),
            })
        present_track_ids = set()
        track_motion = {}
        window_start_sample += window_sample_count

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame_index % stride == 0:
                if sample_index > window_start_sample and sample_index-window_start_sample >= window_samples:
                    finish_window(sample_index-window_start_sample, source_frame_index)
                processed = preprocess_frame(frame)
                detections = _detect(bundle["detector"], bundle["detector_input_name"], processed, frame.shape[:2], class_ids)
                tracks, next_id, associations = _update_tracks(detections, tracks, next_id, sample_index)
                for track_id, detection, distance, elapsed in associations:
                    present_track_ids.add(track_id)
                    if elapsed > 0:
                        motion = track_motion.setdefault(track_id, [0.0, 0.0])
                        motion[0] += distance
                        motion[1] += elapsed
                sample_index += 1
            source_frame_index += 1
            if progress_callback and frame_count > 0:
                pct = min(99, int(source_frame_index * 100 / frame_count))
                if pct > last_progress:
                    last_progress = pct
                    progress_callback(source_frame_index, frame_count)
    finally:
        capture.release()
    if source_frame_index == 0 or sample_index == 0:
        raise ValueError("Video không có frame đọc được")
    remaining = sample_index-window_start_sample
    if remaining > 0:
        finish_window(remaining, source_frame_index - 1)
    if progress_callback:
        progress_callback(source_frame_index, max(source_frame_index, frame_count))
    return {
        "status": "success",
        "pipeline": "AICC video inference",
        "model_version": bundle["manifest"].get("version", "unknown"),
        "video": {
            "frames_processed": source_frame_index,
            "frames_sampled": sample_index,
            "frame_stride": stride,
            "detector_sample_fps": round(sample_fps, 3),
            "fps": round(fps, 3),
            "duration_seconds": round(source_frame_index/fps, 3),
            "width": width,
            "height": height,
        },
        "window_seconds": WINDOW_SECONDS,
        "sequence_length": sequence_length,
        "results_count": len(results),
        "note": "Detector lấy mẫu khung hình để giảm tải CPU. Tốc độ là pixel/giây, chưa được hiệu chuẩn sang km/h. AICC v1 phân loại 2 lớp theo cửa sổ video; không phải mô hình CI dự báo +5/+10/+15 phút của đề án.",
        "results": results,
    }



# Shared process-wide runtime: lazy-loads ONNX only when an operator requests inference.
aicc_runtime = AICCRuntime()
