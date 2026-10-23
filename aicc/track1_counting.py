"""AI City Challenge Track 1 Vehicle Counting Pipeline.

Provides complete detection -> tracking -> ROI verification -> MOI movement assignment -> count-on-exit
pipeline using the pre-existing YOLOv8n detector ONNX model and AI City Challenge Track 1 conventions.
"""
from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment


class Track1VehicleClass(IntEnum):
    """Vehicle classes specified by AI City Challenge Track 1."""
    CAR = 1    # Four-wheel vehicles (cars, sedans, SUVs, pickups, vans)
    TRUCK = 2  # Commercial freight trucks, large heavy vehicles, buses


# Mapping from standard COCO 80 classes to Track 1 classes
COCO_TO_TRACK1 = {
    2: Track1VehicleClass.CAR,    # COCO 'car'
    7: Track1VehicleClass.TRUCK,  # COCO 'truck'
    5: Track1VehicleClass.TRUCK,  # COCO 'bus' (grouped into Track 1 Truck category)
}


@dataclass
class Track1Detection:
    """Single vehicle detection box."""
    box: Tuple[float, float, float, float]  # (x1, y1, w, h)
    score: float
    coco_class_id: int
    track1_class_id: int
    center: Tuple[float, float]


@dataclass
class MOIDefinition:
    """Definition of a Movement of Interest (MOI)."""
    movement_id: int
    name: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    direction_vector: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            self.direction_vector = (dx / norm, dy / norm)
        else:
            self.direction_vector = (0.0, 0.0)


# Calibrated reference movements for camera 5 (from screen_shot_with_roi_and_movement/cam_5.jpg and road geometry)
CAM5_MOVEMENTS = [
    # Approach 1 (North-West / Top-Left)
    MOIDefinition(1, "Appr1_RightTurn", (115.0, 260.0), (45.0, 370.0)),
    MOIDefinition(2, "Appr1_Through", (180.0, 225.0), (850.0, 670.0)),
    MOIDefinition(3, "Appr1_LeftTurn", (240.0, 220.0), (900.0, 170.0)),
    # Approach 2 (North-East / Top-Right)
    MOIDefinition(4, "Appr2_RightTurn", (560.0, 140.0), (260.0, 160.0)),
    MOIDefinition(5, "Appr2_Through", (650.0, 140.0), (45.0, 370.0)),
    MOIDefinition(6, "Appr2_LeftTurn", (675.0, 160.0), (650.0, 700.0)),
    # Approach 3 (East / Right)
    MOIDefinition(7, "Appr3_RightTurn", (900.0, 315.0), (920.0, 180.0)),
    MOIDefinition(8, "Appr3_Through", (990.0, 460.0), (250.0, 180.0)),
    MOIDefinition(9, "Appr3_LeftTurn", (965.0, 535.0), (100.0, 750.0)),
    # Approach 4 (South-West / Bottom-Left)
    MOIDefinition(10, "Appr4_RightTurn", (100.0, 835.0), (700.0, 830.0)),
    MOIDefinition(11, "Appr4_Through", (60.0, 670.0), (900.0, 170.0)),
    MOIDefinition(12, "Appr4_LeftTurn", (50.0, 580.0), (250.0, 200.0)),
]


def load_roi_polygon(roi_path: Path | str) -> np.ndarray:
    """Parse comma-separated polygon vertices from ROI txt file."""
    path = Path(roi_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROI file not found: {path}")
    vertices = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            vertices.append([float(parts[0]), float(parts[1])])
    if len(vertices) < 3:
        raise ValueError(f"Invalid ROI polygon with fewer than 3 vertices in {path}")
    return np.array(vertices, dtype=np.int32)


class Track1VehicleTracker:
    """Multi-object vehicle tracker with trajectory, ROI history and exit event management."""

    def __init__(self, track_id: int, initial_det: Track1Detection, initial_frame: int, in_roi: bool):
        self.track_id = track_id
        self.points: List[Tuple[float, float]] = [initial_det.center]
        self.boxes: List[Tuple[float, float, float, float]] = [initial_det.box]
        self.frames: List[int] = [initial_frame]
        self.class_votes: List[int] = [initial_det.track1_class_id]
        self.in_roi_history: List[bool] = [in_roi]
        self.counted: bool = False
        self.exit_frame: Optional[int] = None
        self.assigned_movement: Optional[int] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    def predict_next_point(self, current_frame: int) -> Tuple[float, float]:
        dt = current_frame - self.frames[-1]
        last_x, last_y = self.points[-1]
        vx, vy = self.velocity
        return (last_x + vx * dt, last_y + vy * dt)

    def update(self, det: Track1Detection, frame_idx: int, in_roi: bool) -> None:
        dt = max(1, frame_idx - self.frames[-1])
        cur_vx = (det.center[0] - self.points[-1][0]) / dt
        cur_vy = (det.center[1] - self.points[-1][1]) / dt
        alpha = 0.6
        self.velocity = (
            alpha * cur_vx + (1.0 - alpha) * self.velocity[0],
            alpha * cur_vy + (1.0 - alpha) * self.velocity[1],
        )
        self.points.append(det.center)
        self.boxes.append(det.box)
        self.frames.append(frame_idx)
        self.class_votes.append(det.track1_class_id)
        self.in_roi_history.append(in_roi)

    @property
    def majority_class(self) -> int:
        car_count = sum(1 for c in self.class_votes if c == Track1VehicleClass.CAR)
        truck_count = sum(1 for c in self.class_votes if c == Track1VehicleClass.TRUCK)
        return int(Track1VehicleClass.CAR if car_count >= truck_count else Track1VehicleClass.TRUCK)

    @property
    def travel_distance(self) -> float:
        if len(self.points) < 2:
            return 0.0
        p_start = self.points[0]
        p_end = self.points[-1]
        return float(math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))

    @property
    def roi_frames_count(self) -> int:
        return sum(1 for in_r in self.in_roi_history if in_r)


TRAJECTORY_FEATURE_NAMES = [
    "x_start_norm",
    "y_start_norm",
    "x_end_norm",
    "y_end_norm",
    "dx_norm",
    "dy_norm",
    "travel_dist_norm",
    "cos_angle",
    "sin_angle",
    "duration_frames",
    "avg_speed_norm",
    "x_mid_norm",
    "y_mid_norm",
    "mean_box_w_norm",
    "mean_box_h_norm",
]


def extract_trajectory_features(track: Track1VehicleTracker, img_w: float = 1280.0, img_h: float = 960.0) -> List[float]:
    """Compute normalized 15-dimensional kinematic and geometric feature vector from a vehicle trajectory."""
    points = track.points
    boxes = track.boxes
    frames = track.frames

    if len(points) < 1:
        return [0.0] * len(TRAJECTORY_FEATURE_NAMES)

    p_start = points[0]
    p_end = points[-1]
    mid_idx = len(points) // 2
    p_mid = points[mid_idx]

    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) if dist > 1e-4 else 0.0

    duration = max(1, frames[-1] - frames[0])
    avg_speed = (dist / duration) / math.hypot(img_w, img_h)

    mean_bw = float(np.mean([b[2] for b in boxes])) if boxes else 0.0
    mean_bh = float(np.mean([b[3] for b in boxes])) if boxes else 0.0

    return [
        round(p_start[0] / img_w, 5),
        round(p_start[1] / img_h, 5),
        round(p_end[0] / img_w, 5),
        round(p_end[1] / img_h, 5),
        round(dx / img_w, 5),
        round(dy / img_h, 5),
        round(dist / math.hypot(img_w, img_h), 5),
        round(math.cos(angle), 5),
        round(math.sin(angle), 5),
        float(duration),
        round(avg_speed, 6),
        round(p_mid[0] / img_w, 5),
        round(p_mid[1] / img_h, 5),
        round(mean_bw / img_w, 5),
        round(mean_bh / img_h, 5),
    ]


class Track1MovementMatcher:
    """Matches a vehicle trajectory to the most similar Movement of Interest."""

    def __init__(
        self,
        movements: List[MOIDefinition],
        ml_model_path: Optional[Path | str] = None,
        classifier_type: str = "heuristic",
    ):
        self.movements = movements
        self.classifier_type = classifier_type
        self.ml_checkpoint: Optional[Dict[str, Any]] = None
        self.ml_model: Any = None
        self.ml_scaler: Any = None
        self.cnn_lstm_ckpt: Optional[Dict[str, Any]] = None
        self.model_type: str = "heuristic"
        if ml_model_path is not None:
            self.load_ml_model(ml_model_path)

    def load_ml_model(self, model_path: Path | str) -> bool:
        """Load trained machine learning movement classifier checkpoint (MLP or CNN-LSTM)."""
        p = Path(model_path)
        if not p.is_file():
            return False
        try:
            with open(p, "rb") as f:
                ckpt = pickle.load(f)
            self.ml_checkpoint = ckpt
            model_type_tag = ckpt.get("model_type", "")
            if model_type_tag == "CNN_LSTM_SEQUENCE_CLASSIFIER":
                self.cnn_lstm_ckpt = ckpt
                self.model_type = "cnn_lstm"
            else:
                self.ml_model = ckpt.get("model")
                self.ml_scaler = ckpt.get("scaler")
                self.model_type = "mlp"
            return True
        except Exception:
            return False

    def _predict_cnn_lstm(self, track: Track1VehicleTracker) -> Optional[int]:
        """Sequence inference using trained CNN-LSTM weights with variable sequence masking."""
        if self.cnn_lstm_ckpt is None or len(track.points) < 2:
            return None
        try:
            max_seq_len = 20
            points = track.points
            boxes = track.boxes
            frames = track.frames
            T = min(len(points), len(boxes))
            if T == 0:
                return None

            if T > max_seq_len:
                indices = np.linspace(0, T - 1, max_seq_len, dtype=int)
            else:
                indices = np.arange(T)

            feat_dim = self.cnn_lstm_ckpt.get("feature_dim", 25)
            spatial_dim = self.cnn_lstm_ckpt.get("spatial_dim", 16)
            seq_feats = np.zeros((len(indices), feat_dim), dtype=np.float32)

            img_w, img_h = 1280.0, 960.0
            for step_i, idx in enumerate(indices):
                cx, cy = points[idx]
                bx, by, bw, bh = boxes[idx]
                if idx > 0:
                    dt = max(1, frames[idx] - frames[idx - 1])
                    dx = points[idx][0] - points[idx - 1][0]
                    dy = points[idx][1] - points[idx - 1][1]
                else:
                    dt, dx, dy = 1, 0.0, 0.0
                dist = math.hypot(dx, dy)
                speed = (dist / dt) / math.hypot(img_w, img_h)
                angle = math.atan2(dy, dx) if dist > 1e-4 else 0.0

                kin = [
                    round(cx / img_w, 5),
                    round(cy / img_h, 5),
                    round(bw / img_w, 5),
                    round(bh / img_h, 5),
                    round(dx / img_w, 5),
                    round(dy / img_h, 5),
                    round(speed, 6),
                    round(math.cos(angle), 5),
                    round(math.sin(angle), 5),
                ]

                norm_cx = cx / img_w
                norm_cy = cy / img_h
                norm_w = bw / img_w
                norm_h = bh / img_h
                aspect = bw / max(1.0, bh)
                spatial = [
                    norm_cx, norm_cy, norm_w, norm_h,
                    aspect, norm_cx**2, norm_cy**2, norm_w * norm_h,
                    math.sin(norm_cx * math.pi), math.cos(norm_cy * math.pi),
                    speed * norm_cx, speed * norm_cy,
                    norm_cx - 0.5, norm_cy - 0.5,
                    abs(norm_cx - 0.5), abs(norm_cy - 0.5),
                ][:spatial_dim]

                seq_feats[step_i] = np.concatenate([spatial, kin])

            mean = self.cnn_lstm_ckpt["feat_mean"]
            scale = self.cnn_lstm_ckpt["feat_scale"]
            norm_seq = (seq_feats - mean) / scale

            H = self.cnn_lstm_ckpt["hidden_dim"]
            W = self.cnn_lstm_ckpt["lstm_W"]
            U = self.cnn_lstm_ckpt["lstm_U"]
            b = self.cnn_lstm_ckpt["lstm_b"]

            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            for t in range(len(indices)):
                xt = norm_seq[t]
                gates = W @ xt + U @ h + b
                i_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[0:H], -15.0, 15.0)))
                f_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[H : 2 * H], -15.0, 15.0)))
                g_gate = np.tanh(gates[2 * H : 3 * H])
                o_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[3 * H : 4 * H], -15.0, 15.0)))
                c = f_gate * c + i_gate * g_gate
                h = o_gate * np.tanh(c)

            W_out = self.cnn_lstm_ckpt["W_out"]
            b_out = self.cnn_lstm_ckpt["b_out"]
            logits = W_out @ h + b_out
            e_s = np.exp(logits - np.max(logits))
            probs = e_s / np.sum(e_s)

            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            class_mapping = self.cnn_lstm_ckpt["class_mapping"]
            pred_mov = class_mapping.get(cls_idx)

            if conf >= 0.40 and pred_mov is not None and 1 <= pred_mov <= 12:
                return int(pred_mov)
            return None
        except Exception:
            return None

    def match_heuristic(self, track: Track1VehicleTracker) -> int:
        """Calibrated geometric heuristic matching."""
        if not self.movements or len(track.points) < 2:
            return 1

        p_start = track.points[0]
        p_end = track.points[-1]
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        traj_norm = math.hypot(dx, dy)
        if traj_norm < 1e-4:
            return 1
        traj_vec = (dx / traj_norm, dy / traj_norm)

        best_mov_id = self.movements[0].movement_id
        best_score = -1e9

        for mov in self.movements:
            cos_sim = traj_vec[0] * mov.direction_vector[0] + traj_vec[1] * mov.direction_vector[1]
            d_start = math.hypot(p_start[0] - mov.start_point[0], p_start[1] - mov.start_point[1]) / 1000.0
            d_end = math.hypot(p_end[0] - mov.end_point[0], p_end[1] - mov.end_point[1]) / 1000.0
            score = 2.5 * cos_sim - 1.2 * d_start - 0.8 * d_end
            if score > best_score:
                best_score = score
                best_mov_id = mov.movement_id

        return best_mov_id

    def match(self, track: Track1VehicleTracker) -> int:
        """Predict movement using ML/CNN-LSTM if available, falling back to geometric heuristic."""
        if self.model_type == "cnn_lstm":
            pred_lstm = self._predict_cnn_lstm(track)
            if pred_lstm is not None:
                return pred_lstm
        elif self.model_type == "mlp" and self.ml_model is not None and len(track.points) >= 2:
            try:
                feats = extract_trajectory_features(track)
                x_arr = np.array([feats], dtype=np.float32)
                if self.ml_scaler is not None:
                    x_arr = self.ml_scaler.transform(x_arr)
                if hasattr(self.ml_model, "predict_proba"):
                    probs = self.ml_model.predict_proba(x_arr)[0]
                    c_idx = int(np.argmax(probs))
                    conf = float(probs[c_idx])
                    pred_cls = int(self.ml_model.classes_[c_idx])
                    if conf >= 0.35 and 1 <= pred_cls <= 12:
                        return pred_cls
                else:
                    pred_cls = int(self.ml_model.predict(x_arr)[0])
                    if 1 <= pred_cls <= 12:
                        return pred_cls
            except Exception:
                pass
        return self.match_heuristic(track)


@dataclass
class Track1CountingEvent:
    """Represents a single counted vehicle exit event matching Track 1 schema."""
    video_clip_id: int
    frame_id: int
    movement_id: int
    vehicle_class_id: int
    track_id: int


class Track1CountingPipeline:
    """End-to-end Track 1 inference pipeline: detect -> track -> check ROI -> count on exit."""

    def __init__(
        self,
        detector_session: ort.InferenceSession,
        roi_polygon: np.ndarray,
        movements: List[MOIDefinition],
        video_clip_id: int = 10,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_track_age: int = 10,
        min_travel_distance: float = 35.0,
        min_roi_frames: int = 3,
        ml_model_path: Optional[Path | str] = None,
    ):
        self.session = detector_session
        self.input_name = detector_session.get_inputs()[0].name
        self.roi = roi_polygon
        self.movements = movements
        self.matcher = Track1MovementMatcher(movements, ml_model_path=ml_model_path)
        self.video_clip_id = video_clip_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_track_age = max_track_age
        self.min_travel_distance = min_travel_distance
        self.min_roi_frames = min_roi_frames

        self.active_tracks: Dict[int, Track1VehicleTracker] = {}
        self.finished_tracks: List[Track1VehicleTracker] = []
        self.next_track_id: int = 1
        self.counting_events: List[Track1CountingEvent] = []

    def _detect(self, frame: np.ndarray) -> List[Track1Detection]:
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0].squeeze()
        boxes = out[:4, :].T
        scores = out[4:, :].T
        classes = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        valid_idx = np.where((max_scores >= self.conf_threshold) & np.isin(classes, list(COCO_TO_TRACK1.keys())))[0]
        if len(valid_idx) == 0:
            return []

        scaled_boxes = []
        det_scores = []
        det_coco_cls = []
        for idx in valid_idx:
            cx, cy, w, h = boxes[idx]
            x1 = (cx - w / 2.0) * (orig_w / 640.0)
            y1 = (cy - h / 2.0) * (orig_h / 640.0)
            bw = w * (orig_w / 640.0)
            bh = h * (orig_h / 640.0)
            scaled_boxes.append([x1, y1, bw, bh])
            det_scores.append(float(max_scores[idx]))
            det_coco_cls.append(int(classes[idx]))

        indices = cv2.dnn.NMSBoxes(scaled_boxes, det_scores, self.conf_threshold, self.iou_threshold)
        detections: List[Track1Detection] = []
        if len(indices) == 0:
            return []

        for i in indices:
            b = scaled_boxes[i]
            c = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
            coco_cls = det_coco_cls[i]
            t1_cls = int(COCO_TO_TRACK1[coco_cls])
            detections.append(Track1Detection(
                box=(b[0], b[1], b[2], b[3]),
                score=det_scores[i],
                coco_class_id=coco_cls,
                track1_class_id=t1_cls,
                center=c,
            ))
        return detections

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        detections = self._detect(frame)

        # Classify detections against ROI
        det_in_roi = [cv2.pointPolygonTest(self.roi, det.center, False) >= 0 for det in detections]

        active_tids = list(self.active_tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if len(active_tids) > 0 and len(detections) > 0:
            cost_matrix = np.full((len(active_tids), len(detections)), 1e6, dtype=np.float32)
            for i, tid in enumerate(active_tids):
                pred_pt = self.active_tracks[tid].predict_next_point(frame_idx)
                for j, det in enumerate(detections):
                    d = math.hypot(det.center[0] - pred_pt[0], det.center[1] - pred_pt[1])
                    if d <= 85.0:
                        cost_matrix[i, j] = d

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 85.0:
                    tid = active_tids[r]
                    det = detections[c]
                    in_roi = det_in_roi[c]
                    track = self.active_tracks[tid]
                    was_in_roi = track.in_roi_history[-1]

                    # Check if vehicle just exited the ROI
                    if was_in_roi and not in_roi and not track.counted:
                        track.update(det, frame_idx, in_roi)
                        self._check_and_emit_exit(track, frame_idx)
                    else:
                        track.update(det, frame_idx, in_roi)

                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Drop expired tracks
        for tid in active_tids:
            if tid not in matched_tracks:
                track = self.active_tracks[tid]
                if frame_idx - track.frames[-1] > self.max_track_age:
                    dist_to_boundary = cv2.pointPolygonTest(self.roi, track.points[-1], True)
                    # Emit exit if vehicle reached near boundary (<= 90px) or traversed significant journey (>= 120px)
                    if dist_to_boundary <= 90.0 or track.travel_distance >= 120.0:
                        self._check_and_emit_exit(track, track.frames[-1])
                    del self.active_tracks[tid]

        # Initialize new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                in_roi = det_in_roi[d_idx]
                self.active_tracks[self.next_track_id] = Track1VehicleTracker(
                    track_id=self.next_track_id,
                    initial_det=det,
                    initial_frame=frame_idx,
                    in_roi=in_roi,
                )
                next_track_id_cand = self.next_track_id + 1
                self.next_track_id = next_track_id_cand

    def _check_and_emit_exit(self, track: Track1VehicleTracker, exit_frame: int) -> None:
        if track.counted:
            return
        # Eligibility criteria:
        # 1. Must have appeared inside ROI for at least min_roi_frames
        # 2. Must have traveled a minimum distance across the scene
        if track.roi_frames_count >= self.min_roi_frames and track.travel_distance >= self.min_travel_distance:
            mov_id = self.matcher.match(track)
            cls_id = track.majority_class
            event = Track1CountingEvent(
                video_clip_id=self.video_clip_id,
                frame_id=exit_frame,
                movement_id=mov_id,
                vehicle_class_id=cls_id,
                track_id=track.track_id,
            )
            self.counting_events.append(event)
            track.counted = True
            track.exit_frame = exit_frame
            track.assigned_movement = mov_id
            self.finished_tracks.append(track)

    def finalize(self, last_frame_idx: int) -> List[Track1CountingEvent]:
        """Flush remaining active tracks at the end of the video."""
        for tid, track in list(self.active_tracks.items()):
            dist_to_boundary = cv2.pointPolygonTest(self.roi, track.points[-1], True)
            if dist_to_boundary <= 90.0 or track.travel_distance >= 120.0:
                self._check_and_emit_exit(track, last_frame_idx)
        return self.counting_events


def export_track1_csv(events: List[Track1CountingEvent], output_path: Path | str) -> Path:
    """Export counting results strictly matching the official AI City Track 1 CSV schema."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_events = sorted(events, key=lambda e: (e.video_clip_id, e.frame_id, e.movement_id))
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_clip_id", "frame_id", "movement_id", "vehicle_class_id"])
        for e in sorted_events:
            writer.writerow([e.video_clip_id, e.frame_id, e.movement_id, e.vehicle_class_id])
    return out_file


def evaluate_track1_predictions(
    predicted_csv: Path | str,
    ground_truth_csv: Path | str,
    tolerance_frames: int = 15,
) -> Dict[str, Any]:
    """Evaluate predicted counting CSV against ground truth CSV.
    
    Computes total counts, per-movement breakdown, per-class breakdown, MAE, RMSE,
    and event matching rate within frame tolerance window.
    """
    pred_path = Path(predicted_csv)
    gt_path = Path(ground_truth_csv)

    def load_rows(p: Path) -> List[Dict[str, int]]:
        with open(p, "r", encoding="utf-8-sig") as f:
            return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]

    pred_rows = load_rows(pred_path)
    gt_rows = load_rows(gt_path)

    # Class counts
    gt_car = sum(1 for r in gt_rows if r["vehicle_class_id"] == 1)
    gt_truck = sum(1 for r in gt_rows if r["vehicle_class_id"] == 2)
    pred_car = sum(1 for r in pred_rows if r["vehicle_class_id"] == 1)
    pred_truck = sum(1 for r in pred_rows if r["vehicle_class_id"] == 2)

    # Movement breakdown across all active and defined movements
    all_movements = sorted(set([r["movement_id"] for r in gt_rows] + [r["movement_id"] for r in pred_rows]))
    movement_breakdown = {}
    mae_sum = 0.0
    se_sum = 0.0

    for m in range(1, 13):
        gt_m = sum(1 for r in gt_rows if r["movement_id"] == m)
        pred_m = sum(1 for r in pred_rows if r["movement_id"] == m)
        diff = pred_m - gt_m
        mae_sum += abs(diff)
        se_sum += diff ** 2
        movement_breakdown[str(m)] = {
            "ground_truth": gt_m,
            "predicted": pred_m,
            "error": diff,
        }

    # MAE/RMSE across active movements (present in video sample, N=10) and all movements (N=12)
    num_active_movs = max(1, len(all_movements))
    active_mae_sum = sum(abs(movement_breakdown[str(m)]["error"]) for m in all_movements)
    active_se_sum = sum(movement_breakdown[str(m)]["error"] ** 2 for m in all_movements)
    mae_active = round(active_mae_sum / num_active_movs, 3)
    rmse_active = round(math.sqrt(active_se_sum / num_active_movs), 3)

    mae_12_all = round(mae_sum / 12.0, 3)
    rmse_12_all = round(math.sqrt(se_sum / 12.0), 3)

    # MAE/RMSE across 24 cells (12 movements x 2 vehicle classes)
    cell_mae_sum = 0.0
    cell_se_sum = 0.0
    movement_class_breakdown = {}
    for m in range(1, 13):
        for cls_id, cls_name in [(1, "car"), (2, "truck")]:
            gt_mc = sum(1 for r in gt_rows if r["movement_id"] == m and r["vehicle_class_id"] == cls_id)
            pred_mc = sum(1 for r in pred_rows if r["movement_id"] == m and r["vehicle_class_id"] == cls_id)
            err_mc = pred_mc - gt_mc
            cell_mae_sum += abs(err_mc)
            cell_se_sum += err_mc ** 2
            if gt_mc > 0 or pred_mc > 0:
                movement_class_breakdown[f"{m}_{cls_name}"] = {
                    "movement_id": m,
                    "class": cls_name,
                    "ground_truth": gt_mc,
                    "predicted": pred_mc,
                    "error": err_mc,
                }
    mae_24_cells = round(cell_mae_sum / 24.0, 3)
    rmse_24_cells = round(math.sqrt(cell_se_sum / 24.0), 3)

    # Pass 1: Optimal Bipartite Match for Strict Temporal Matching (movement + class + time window)
    tp = 0
    matched_gt_set = set()
    matched_pred_set = set()
    if len(gt_rows) > 0 and len(pred_rows) > 0:
        cost_strict = np.full((len(gt_rows), len(pred_rows)), 1e6, dtype=np.float32)
        for g_idx, g in enumerate(gt_rows):
            for p_idx, p in enumerate(pred_rows):
                if p["movement_id"] == g["movement_id"] and p["vehicle_class_id"] == g["vehicle_class_id"]:
                    diff = abs(p["frame_id"] - g["frame_id"])
                    if diff <= tolerance_frames:
                        cost_strict[g_idx, p_idx] = diff
        r_ind, c_ind = linear_sum_assignment(cost_strict)
        for r, c in zip(r_ind, c_ind):
            if cost_strict[r, c] <= tolerance_frames:
                tp += 1
                matched_gt_set.add(r)
                matched_pred_set.add(c)

    fp = len(pred_rows) - tp
    fn = len(gt_rows) - tp
    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / max(1, tp + fn), 4)
    f1 = round(2 * tp / max(1e-6, 2 * tp + fp + fn), 4)

    # Pass 2: Count-Only Temporal Matching (reconciled by including verified Wrong Movement events)
    unmatched_gt = [i for i in range(len(gt_rows)) if i not in matched_gt_set]
    unmatched_pred = [j for j in range(len(pred_rows)) if j not in matched_pred_set]
    wm_count = 0
    if len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
        cost_wm = np.full((len(unmatched_gt), len(unmatched_pred)), 1e6, dtype=np.float32)
        for i, g_idx in enumerate(unmatched_gt):
            g = gt_rows[g_idx]
            for j, p_idx in enumerate(unmatched_pred):
                p = pred_rows[p_idx]
                if p["vehicle_class_id"] == g["vehicle_class_id"] and p["movement_id"] != g["movement_id"]:
                    diff = abs(p["frame_id"] - g["frame_id"])
                    if diff <= tolerance_frames:
                        cost_wm[i, j] = diff
        r_wm, c_wm = linear_sum_assignment(cost_wm)
        wm_count = sum(1 for r, c in zip(r_wm, c_wm) if cost_wm[r, c] <= tolerance_frames)

    count_only_tp = tp + wm_count
    count_only_fp = len(pred_rows) - count_only_tp
    count_only_fn = len(gt_rows) - count_only_tp
    count_only_prec = round(count_only_tp / max(1, len(pred_rows)), 4)
    count_only_rec = round(count_only_tp / max(1, len(gt_rows)), 4)
    count_only_f1 = round(2 * count_only_tp / max(1e-6, 2 * count_only_tp + count_only_fp + count_only_fn), 4)

    return {
        "status": "success",
        "total_ground_truth": len(gt_rows),
        "total_predicted": len(pred_rows),
        "ground_truth_classes": {"car": gt_car, "truck": gt_truck},
        "predicted_classes": {"car": pred_car, "truck": pred_truck},
        "overall_count_difference": len(pred_rows) - len(gt_rows),
        "mae_per_movement": mae_active,
        "rmse_per_movement": rmse_active,
        "mae_all_12_movements": mae_12_all,
        "rmse_all_12_movements": rmse_12_all,
        "mae_24_movement_class_cells": mae_24_cells,
        "rmse_24_movement_class_cells": rmse_24_cells,
        "temporal_matching_tolerance_frames": tolerance_frames,
        "matched_events_tp": tp,
        "unmatched_predictions_fp": fp,
        "missed_gt_events_fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "count_only_evaluation": {
            "matched_tp": count_only_tp,
            "unmatched_fp": count_only_fp,
            "missed_fn": count_only_fn,
            "precision": count_only_prec,
            "recall": count_only_rec,
            "f1_score": count_only_f1,
        },
        "movement_breakdown": movement_breakdown,
        "movement_class_breakdown": movement_class_breakdown,
    }
