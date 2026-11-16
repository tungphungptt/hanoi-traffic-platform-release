"""ONNX vehicle detector (COCO classes) and a greedy nearest-centre tracker, on CPU.

Shipped: YOLOv8n (AGPL-3.0). Also understood, picked from the model's output shape: the official YOLOX
ONNX exports (Apache-2.0), e.g. ``DETECTOR_MODEL=/path/yolox_s.onnx``. Changing the detector changes the
CI scale, so the forecast library must be rebuilt with the same detector.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_PATH = Path(os.getenv("DETECTOR_MODEL") or Path(__file__).resolve().parents[1] / "models" / "detector.onnx")
INPUT_SIZE = 640
VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})  # COCO: car, motorcycle, bus, truck
PERSON_CLASS_ID = 0  # only used to blur people in saved pictures, never counted as traffic
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
TRACK_DISTANCE_PIXELS = 50.0
MAX_TRACK_AGE = 5  # samples


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
    last_sample: int


def parse_detections(output: np.ndarray, input_shape: tuple[int, int], frame_shape: tuple[int, int],
                     class_ids: frozenset[int] = VEHICLE_CLASS_IDS,
                     min_score: float = CONFIDENCE_THRESHOLD) -> list[Detection]:
    """Decode a YOLOv8 output ([N,6] xyxy/score/class or [4+classes, N] cxcywh) and apply NMS."""
    out = np.asarray(output).squeeze()
    if out.ndim != 2:
        raise RuntimeError(f"Unexpected detector output shape: {out.shape}")
    if out.shape[0] < out.shape[1] and out.shape[0] != 6:
        out = out.T
    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    classes: list[int] = []
    if out.shape[1] == 6:
        for row in out:
            class_id, score = int(row[5]), float(row[4])
            if class_id in class_ids and score >= min_score:
                boxes.append(tuple(map(float, row[:4])))
                scores.append(score)
                classes.append(class_id)
    elif out.shape[1] > 6:
        class_scores = out[:, 4:]
        best = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(len(out)), best]
        for row, class_id, score in zip(out, best, best_score):
            if int(class_id) not in class_ids or score < min_score:
                continue
            cx, cy, w, h = map(float, row[:4])
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            scores.append(float(score))
            classes.append(int(class_id))
    else:
        raise RuntimeError(f"Unexpected detector output shape: {out.shape}")
    kept: list[int] = []
    for class_id in set(classes):
        idx = [i for i, c in enumerate(classes) if c == class_id]
        xywh = [[boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in idx]
        nms = cv2.dnn.NMSBoxes(xywh, [scores[i] for i in idx], min_score, NMS_THRESHOLD)
        kept.extend(idx[int(i)] for i in np.asarray(nms).reshape(-1))
    (ih, iw), (fh, fw) = input_shape, frame_shape
    sx, sy = fw / iw, fh / ih
    return [Detection(classes[i], (boxes[i][0] * sx, boxes[i][1] * sy, boxes[i][2] * sx, boxes[i][3] * sy), scores[i])
            for i in sorted(kept)]


YOLOX_STRIDES = (8, 16, 32)


def yolox_letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize keeping the aspect ratio, pad bottom/right with 114; returns (NCHW float32 BGR 0-255, ratio)."""
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def parse_yolox(output: np.ndarray, size: int, ratio: float, class_ids: frozenset[int] = VEHICLE_CLASS_IDS,
                min_score: float = CONFIDENCE_THRESHOLD) -> list[Detection]:
    """Decode the raw YOLOX head ([N, 85]: cx, cy, w, h as grid offsets, objectness, 80 class scores) and apply NMS."""
    out = np.asarray(output).reshape(-1, np.asarray(output).shape[-1])
    grids, strides = [], []
    for stride in YOLOX_STRIDES:
        n = size // stride
        gx, gy = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((gx, gy), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    if len(grid) != len(out):
        raise RuntimeError(f"Unexpected YOLOX output rows: {len(out)} for input {size}")
    centre = (out[:, :2] + grid) * stride
    wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * stride
    scores_all = out[:, 4:5] * out[:, 5:]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(out)), cls]
    keep = np.isin(cls, list(class_ids)) & (score >= min_score)
    boxes = (np.concatenate([centre - wh / 2, centre + wh / 2], 1)[keep]) / ratio
    cls, score = cls[keep], score[keep]
    found: list[Detection] = []
    for class_id in set(cls.tolist()):
        idx = np.flatnonzero(cls == class_id)
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[idx]]
        nms = cv2.dnn.NMSBoxes(xywh, score[idx].tolist(), min_score, NMS_THRESHOLD)
        for k in np.asarray(nms).reshape(-1):
            b = boxes[idx[int(k)]]
            found.append(Detection(int(class_id), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(score[idx[int(k)]])))
    return found


class Detector:
    """Loads the ONNX model (models/detector.onnx, or DETECTOR_MODEL) on first use; shared by the worker thread."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = Path(path)
        self._session: Any = None
        self._input_name = ""
        self._input_size = INPUT_SIZE
        self.kind = "yolov8"  # or "yolox", from the output shape
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if not self.available:
                self.error = f"Thiếu {self.path.name} trong models/."
                raise FileNotFoundError(self.error)
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self._session = ort.InferenceSession(str(self.path), sess_options=options,
                                                 providers=["CPUExecutionProvider"])
            model_input = self._session.get_inputs()[0]
            self._input_name = model_input.name
            if isinstance(model_input.shape[-1], int):
                self._input_size = int(model_input.shape[-1])
            out_shape = self._session.get_outputs()[0].shape
            self.kind = "yolox" if len(out_shape) == 3 and out_shape[-1] == 85 else "yolov8"
            self.error = None

    def detect(self, frame: np.ndarray, class_ids: frozenset[int] = VEHICLE_CLASS_IDS,
               min_score: float = CONFIDENCE_THRESHOLD) -> list[Detection]:
        if self._session is None:
            self.load()
        if self.kind == "yolox":
            tensor, ratio = yolox_letterbox(frame, self._input_size)
            output = self._session.run(None, {self._input_name: tensor})[0]
            return parse_yolox(output, self._input_size, ratio, class_ids, min_score)
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
        output = self._session.run(None, {self._input_name: tensor})[0]
        return parse_detections(output, (INPUT_SIZE, INPUT_SIZE), frame.shape[:2], class_ids, min_score)


def update_tracks(detections: list[Detection], tracks: dict[int, Track], next_id: int,
                  sample: int) -> tuple[dict[int, Track], int, list[tuple[float, int]]]:
    """Greedy nearest-centre association. Returns (tracks, next_id, [(distance_px, elapsed_samples)])."""
    candidates = []
    for di, det in enumerate(detections):
        for tid, track in tracks.items():
            if track.class_id != det.class_id or sample - track.last_sample > MAX_TRACK_AGE:
                continue
            distance = float(np.hypot(det.center[0] - track.center[0], det.center[1] - track.center[1]))
            if distance <= TRACK_DISTANCE_PIXELS:
                candidates.append((distance, di, tid))
    candidates.sort()
    used_det: set[int] = set()
    used_track: set[int] = set()
    moves: list[tuple[float, int]] = []
    updated: dict[int, Track] = {}
    for distance, di, tid in candidates:
        if di in used_det or tid in used_track:
            continue
        det = detections[di]
        moves.append((distance, sample - tracks[tid].last_sample))
        updated[tid] = Track(tid, det.class_id, det.center, sample)
        used_det.add(di)
        used_track.add(tid)
    for tid, track in tracks.items():
        if tid not in used_track and sample - track.last_sample <= MAX_TRACK_AGE:
            updated[tid] = track
    for di, det in enumerate(detections):
        if di not in used_det:
            updated[next_id] = Track(next_id, det.class_id, det.center, sample)
            next_id += 1
    return updated, next_id, moves
