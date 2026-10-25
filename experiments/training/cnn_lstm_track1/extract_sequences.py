"""Extract spatio-temporal tracklet sequences (streaming frame crops + kinematics) for CNN-LSTM training.

Operates with streaming crop extraction to ensure low RAM consumption (<100MB) without buffering full video frames.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicc.track1_counting import (
    CAM5_MOVEMENTS,
    Track1CountingPipeline,
    Track1VehicleTracker,
    load_roi_polygon,
)


def extract_tracklet_sequence_from_cached_crops(
    track: Track1VehicleTracker,
    cached_crops: List[np.ndarray],
    img_w: float = 1280.0,
    img_h: float = 960.0,
    crop_size: int = 64,
    max_seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extract temporal sequence of crops (max_seq_len, 3, H, W) and kinematics (max_seq_len, 9)."""
    points = track.points
    boxes = track.boxes
    frames = track.frames
    T = min(len(points), len(cached_crops))

    if T == 0:
        return (
            np.zeros((max_seq_len, 3, crop_size, crop_size), dtype=np.float32),
            np.zeros((max_seq_len, 9), dtype=np.float32),
            0,
        )

    # Subsample or sample uniformly if longer than max_seq_len
    if T > max_seq_len:
        indices = np.linspace(0, T - 1, max_seq_len, dtype=int)
    else:
        indices = np.arange(T)

    seq_len = len(indices)
    crops = np.zeros((max_seq_len, 3, crop_size, crop_size), dtype=np.float32)
    kinematics = np.zeros((max_seq_len, 9), dtype=np.float32)

    for step_i, idx in enumerate(indices):
        f_num = frames[idx]
        cx, cy = points[idx]
        bx, by, bw, bh = boxes[idx]

        # Use the streaming-cached crop
        if idx < len(cached_crops):
            crops[step_i] = cached_crops[idx]

        # Compute kinematic features at this step
        if idx > 0:
            dt = max(1, frames[idx] - frames[idx - 1])
            dx = points[idx][0] - points[idx - 1][0]
            dy = points[idx][1] - points[idx - 1][1]
        else:
            dt = 1
            dx, dy = 0.0, 0.0

        dist = math.hypot(dx, dy)
        speed = (dist / dt) / math.hypot(img_w, img_h)
        angle = math.atan2(dy, dx) if dist > 1e-4 else 0.0

        kinematics[step_i] = [
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

    return crops, kinematics, seq_len


def build_cnn_lstm_dataset(
    video_path: Path,
    roi_path: Path,
    detector_path: Path,
    gt_csv_path: Path,
    output_dir: Path,
    max_frames: int = 600,
    crop_size: int = 64,
    max_seq_len: int = 20,
) -> Dict[str, Any]:
    """Execute detection+tracking, stream crops on-the-fly, match ground truth, and export sequence dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    roi = load_roi_polygon(roi_path)
    session = ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
    pipeline = Track1CountingPipeline(
        detector_session=session,
        roi_polygon=roi,
        movements=CAM5_MOVEMENTS,
        video_clip_id=10,
        conf_threshold=0.35,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280.0
    img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 960.0

    print("Step 1: Reading video and extracting detections + tracklet sequences (low-memory streaming)...")
    track_snapshots: Dict[int, Tuple[Track1VehicleTracker, List[np.ndarray]]] = {}
    active_crops: Dict[int, List[np.ndarray]] = {}

    orig_emit = pipeline._check_and_emit_exit

    def intercept_emit(track: Track1VehicleTracker, exit_frame: int):
        snap = copy.deepcopy(track)
        snap.exit_frame = exit_frame
        crops_copy = list(active_crops.get(track.track_id, []))
        track_snapshots[track.track_id] = (snap, crops_copy)
        orig_emit(track, exit_frame)

    pipeline._check_and_emit_exit = intercept_emit

    frame_idx = 0
    t0 = time.time()
    try:
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            pipeline.process_frame(frame, frame_idx)

            # Stream crop extraction for all active tracks updated in this frame
            for tid, track in pipeline.active_tracks.items():
                if track.frames[-1] == frame_idx:
                    bx, by, bw, bh = track.boxes[-1]
                    x1 = max(0, int(bx))
                    y1 = max(0, int(by))
                    x2 = min(frame.shape[1], int(bx + bw))
                    y2 = min(frame.shape[0], int(by + bh))
                    if x2 > x1 + 2 and y2 > y1 + 2:
                        patch = frame[y1:y2, x1:x2]
                        patch_resized = cv2.resize(patch, (crop_size, crop_size))
                        patch_chw = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
                    else:
                        patch_chw = np.zeros((3, crop_size, crop_size), dtype=np.float32)

                    if tid not in active_crops:
                        active_crops[tid] = []
                    active_crops[tid].append(patch_chw)

            # Clean up active_crops for dead tracks not in snapshots
            expired = [t for t in active_crops if t not in pipeline.active_tracks and t not in track_snapshots]
            for t in expired:
                del active_crops[t]

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {frame_idx}/{max_frames} frames ({frame_idx / max(1e-4, elapsed):.1f} fps)...")
    finally:
        cap.release()

    events = pipeline.finalize(frame_idx - 1)
    fps_run = frame_idx / max(1e-4, time.time() - t0)
    print(f"Completed: processed {frame_idx} frames ({fps_run:.1f} fps). Emitted {len(events)} events, captured {len(track_snapshots)} track snapshots.")

    print("\nStep 2: Matching tracks with official counting ground truth sample...")
    with open(gt_csv_path, "r", encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))

    gt_list = [
        {"idx": i, "frame": int(r["frame_id"]), "mov": int(r["movement_id"]), "cls": int(r["vehicle_class_id"])}
        for i, r in enumerate(gt_rows)
    ]
    pred_list = [
        {"idx": i, "track_id": e.track_id, "frame": e.frame_id, "mov": e.movement_id, "cls": e.vehicle_class_id}
        for i, e in enumerate(events)
    ]

    # Exact bipartite matching
    cost_exact = np.full((len(gt_list), len(pred_list)), 1e6, dtype=np.float32)
    for g_i, g in enumerate(gt_list):
        for p_i, p in enumerate(pred_list):
            if p["mov"] == g["mov"] and p["cls"] == g["cls"]:
                d = abs(p["frame"] - g["frame"])
                if d <= 15:
                    cost_exact[g_i, p_i] = d

    r_ind, c_ind = linear_sum_assignment(cost_exact)
    matched_records = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(r_ind, c_ind):
        if cost_exact[r, c] <= 15:
            matched_records.append({
                "track_id": pred_list[c]["track_id"],
                "exit_frame": pred_list[c]["frame"],
                "gt_frame": gt_list[r]["frame"],
                "true_gt_mov": gt_list[r]["mov"],
                "heuristic_mov": pred_list[c]["mov"],
                "cls": pred_list[c]["cls"],
                "match_type": "EXACT_MATCH",
            })
            matched_gt.add(r)
            matched_pred.add(c)

    # Wrong movement bipartite matching
    un_gt = [i for i in range(len(gt_list)) if i not in matched_gt]
    un_pred = [i for i in range(len(pred_list)) if i not in matched_pred]
    if un_gt and un_pred:
        cost_wm = np.full((len(un_gt), len(un_pred)), 1e6, dtype=np.float32)
        for i, g_i in enumerate(un_gt):
            g = gt_list[g_i]
            for j, p_j in enumerate(un_pred):
                p = pred_list[p_j]
                if p["cls"] == g["cls"] and p["mov"] != g["mov"]:
                    d = abs(p["frame"] - g["frame"])
                    if d <= 15:
                        cost_wm[i, j] = d

        r_wm, c_wm = linear_sum_assignment(cost_wm)
        for r, c in zip(r_wm, c_wm):
            if cost_wm[r, c] <= 15:
                matched_records.append({
                    "track_id": pred_list[un_pred[c]]["track_id"],
                    "exit_frame": pred_list[un_pred[c]]["frame"],
                    "gt_frame": gt_list[un_gt[r]]["frame"],
                    "true_gt_mov": gt_list[un_gt[r]]["mov"],
                    "heuristic_mov": pred_list[un_pred[c]]["mov"],
                    "cls": pred_list[un_pred[c]]["cls"],
                    "match_type": "WRONG_MOVEMENT",
                })

    print(f"Total matched tracklet samples: {len(matched_records)} (Exact: {len(matched_gt)}, Wrong Mov: {len(matched_records) - len(matched_gt)})")

    print("\nStep 3: Extracting sequences of crops and kinematics...")
    crops_list = []
    kinematics_list = []
    lengths_list = []
    labels_list = []
    final_records = []

    for rec in matched_records:
        tid = rec["track_id"]
        if tid in track_snapshots:
            track, cached_crops = track_snapshots[tid]
            crops, kin, seq_len = extract_tracklet_sequence_from_cached_crops(
                track=track,
                cached_crops=cached_crops,
                img_w=img_w,
                img_h=img_h,
                crop_size=crop_size,
                max_seq_len=max_seq_len,
            )
            crops_list.append(crops)
            kinematics_list.append(kin)
            lengths_list.append(seq_len)
            labels_list.append(rec["true_gt_mov"])
            rec["seq_len"] = seq_len
            final_records.append(rec)

    crops_arr = np.array(crops_list, dtype=np.float32)          # (N, max_seq_len, 3, crop_size, crop_size)
    kin_arr = np.array(kinematics_list, dtype=np.float32)        # (N, max_seq_len, 9)
    len_arr = np.array(lengths_list, dtype=np.int32)            # (N,)
    label_arr = np.array(labels_list, dtype=np.int32)           # (N,)

    dataset_path = output_dir / "cnn_lstm_dataset.npz"
    np.savez_compressed(
        dataset_path,
        crops=crops_arr,
        kinematics=kin_arr,
        lengths=len_arr,
        labels=label_arr,
    )

    metadata = {
        "total_samples": len(final_records),
        "crops_shape": list(crops_arr.shape),
        "kinematics_shape": list(kin_arr.shape),
        "max_seq_len": max_seq_len,
        "crop_size": crop_size,
        "kinematics_dim": 9,
        "records": final_records,
    }
    meta_path = output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved dataset array to: {dataset_path} ({dataset_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Saved dataset metadata to: {meta_path}")
    return metadata


if __name__ == "__main__":
    video_p = ROOT / "AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4"
    roi_p = ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt"
    det_p = ROOT / "models/detector.onnx"
    gt_p = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    out_d = ROOT / "experiments/training/cnn_lstm_track1"

    build_cnn_lstm_dataset(video_p, roi_p, det_p, gt_p, out_d)
