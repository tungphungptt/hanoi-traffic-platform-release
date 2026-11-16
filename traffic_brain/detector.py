"""YOLOv8n ONNX vehicle detector (COCO classes) and a greedy nearest-centre tracker, on CPU."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "detector.onnx"
INPUT_SIZE = 640
VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})  # COCO: car, motorcycle, bus, truck
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
                     class_ids: frozenset[int] = VEHICLE_CLASS_IDS) -> list[Detection]:
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
            if class_id in class_ids and score >= CONFIDENCE_THRESHOLD:
                boxes.append(tuple(map(float, row[:4])))
                scores.append(score)
                classes.append(class_id)
    elif out.shape[1] > 6:
        class_scores = out[:, 4:]
        best = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(len(out)), best]
        for row, class_id, score in zip(out, best, best_score):
            if int(class_id) not in class_ids or score < CONFIDENCE_THRESHOLD:
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
        nms = cv2.dnn.NMSBoxes(xywh, [scores[i] for i in idx], CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        kept.extend(idx[int(i)] for i in np.asarray(nms).reshape(-1))
    (ih, iw), (fh, fw) = input_shape, frame_shape
    sx, sy = fw / iw, fh / ih
    return [Detection(classes[i], (boxes[i][0] * sx, boxes[i][1] * sy, boxes[i][2] * sx, boxes[i][3] * sy), scores[i])
            for i in sorted(kept)]


class Detector:
    """Loads models/detector.onnx on first use; one instance is shared by the worker thread."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = Path(path)
        self._session: Any = None
        self._input_name = ""
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
            self._input_name = self._session.get_inputs()[0].name
            self.error = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._session is None:
            self.load()
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
        output = self._session.run(None, {self._input_name: tensor})[0]
        return parse_detections(output, (INPUT_SIZE, INPUT_SIZE), frame.shape[:2])


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
