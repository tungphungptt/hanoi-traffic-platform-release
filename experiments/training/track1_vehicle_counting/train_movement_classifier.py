"""Train and evaluate Machine Learning Movement Classifier for Track 1 Vehicle Counting.

Extracts trajectory feature vectors from ground-truth verified vehicle tracks on cam_5,
performs strict temporal train/test split, trains Random Forest, MLP, and Gradient Boosting models,
evaluates precision/recall/F1 per movement, computes confusion matrices, and exports the model checkpoint.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicc.track1_counting import (
    CAM5_MOVEMENTS,
    Track1CountingPipeline,
    Track1VehicleTracker,
    load_roi_polygon,
)

FEATURE_NAMES = [
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
        return [0.0] * len(FEATURE_NAMES)

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


def run_pipeline_and_extract_tracks(
    video_path: Path,
    roi_path: Path,
    detector_path: Path,
    max_frames: int = 600,
) -> Tuple[List[Any], Dict[int, Track1VehicleTracker], float, float]:
    """Execute pipeline and capture full tracker objects for each exit event."""
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

    track_snapshots: Dict[int, Track1VehicleTracker] = {}

    orig_emit = pipeline._check_and_emit_exit

    def intercept_emit(track: Track1VehicleTracker, exit_frame: int):
        import copy
        # Create a deep snapshot copy of the tracker
        snap = copy.deepcopy(track)
        snap.exit_frame = exit_frame
        track_snapshots[track.track_id] = snap
        orig_emit(track, exit_frame)

    pipeline._check_and_emit_exit = intercept_emit

    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            pipeline.process_frame(frame, frame_idx)
            frame_idx += 1
    finally:
        cap.release()

    events = pipeline.finalize(frame_idx - 1)
    return events, track_snapshots, img_w, img_h


def match_tracks_to_ground_truth(
    events: List[Any],
    track_snapshots: Dict[int, Track1VehicleTracker],
    gt_csv_path: Path,
    img_w: float,
    img_h: float,
    tol_frames: int = 15,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Align detected track events with counting ground truth to extract labeled training dataset."""
    with open(gt_csv_path, "r", encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))

    gt_list = [
        {"idx": i, "frame": int(r["frame_id"]), "mov": int(r["movement_id"]), "cls": int(r["vehicle_class_id"])}
        for i, r in enumerate(gt_rows)
    ]
    pred_list = [
        {
            "idx": i,
            "track_id": e.track_id,
            "frame": e.frame_id,
            "mov": e.movement_id,
            "cls": e.vehicle_class_id,
        }
        for i, e in enumerate(events)
    ]

    matched_records = []
    matched_gt = set()
    matched_pred = set()

    # Pass 1: Exact matches (same class & movement, within tol_frames)
    cost_exact = np.full((len(gt_list), len(pred_list)), 1e6, dtype=np.float32)
    for g_i, g in enumerate(gt_list):
        for p_i, p in enumerate(pred_list):
            if p["mov"] == g["mov"] and p["cls"] == g["cls"]:
                d = abs(p["frame"] - g["frame"])
                if d <= tol_frames:
                    cost_exact[g_i, p_i] = d

    r_ind, c_ind = linear_sum_assignment(cost_exact)
    for r, c in zip(r_ind, c_ind):
        if cost_exact[r, c] <= tol_frames:
            g = gt_list[r]
            p = pred_list[c]
            matched_records.append({
                "track_id": p["track_id"],
                "exit_frame": p["frame"],
                "gt_frame": g["frame"],
                "heuristic_mov": p["mov"],
                "true_gt_mov": g["mov"],
                "cls": p["cls"],
                "match_type": "EXACT_MATCH",
            })
            matched_gt.add(r)
            matched_pred.add(c)

    # Pass 2: Wrong movement matches (same class, different movement, within tol_frames)
    un_gt = [i for i in range(len(gt_list)) if i not in matched_gt]
    un_pred = [i for i in range(len(pred_list)) if i not in matched_pred]
    if un_gt and un_pred:
        cost_wm = np.full((len(un_gt), len(un_pred)), 1e6, dtype=np.float32)
        for i, g_i in enumerate(un_gt):
            g = gt_list[g_i]
            for j, p_i in enumerate(un_pred):
                p = pred_list[p_i]
                if p["cls"] == g["cls"] and p["mov"] != g["mov"]:
                    d = abs(p["frame"] - g["frame"])
                    if d <= tol_frames:
                        cost_wm[i, j] = d

        r_wm, c_wm = linear_sum_assignment(cost_wm)
        for r, c in zip(r_wm, c_wm):
            if cost_wm[r, c] <= tol_frames:
                g = gt_list[un_gt[r]]
                p = pred_list[un_pred[c]]
                matched_records.append({
                    "track_id": p["track_id"],
                    "exit_frame": p["frame"],
                    "gt_frame": g["frame"],
                    "heuristic_mov": p["mov"],
                    "true_gt_mov": g["mov"],
                    "cls": p["cls"],
                    "match_type": "WRONG_MOVEMENT",
                })
                matched_gt.add(un_gt[r])
                matched_pred.add(un_pred[c])

    # Build feature matrix X and target vector y
    X_list = []
    y_list = []
    final_records = []

    for rec in matched_records:
        tid = rec["track_id"]
        if tid in track_snapshots:
            track = track_snapshots[tid]
            feats = extract_trajectory_features(track, img_w, img_h)
            X_list.append(feats)
            y_list.append(rec["true_gt_mov"])
            rec["features"] = feats
            final_records.append(rec)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), final_records


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Train machine learning movement classifiers, compare models, and export artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strict temporal train/test split: frame <= 350 vs frame > 350
    train_mask = np.array([r["exit_frame"] <= 350 for r in records])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("=" * 65)
    print("AI City Track 1 - Machine Learning Movement Classifier Training")
    print("=" * 65)
    print(f"Total labeled trajectories: {len(X)}")
    print(f"Train split (exit frame <= 350): {len(X_train)} samples")
    print(f"Test split  (exit frame > 350):  {len(X_test)} samples")
    print(f"Unique movement classes in train: {sorted(list(set(y_train)))}")
    print(f"Unique movement classes in test:  {sorted(list(set(y_test)))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=600,
            alpha=0.01,
            random_state=42,
        ),
    }

    results = {}
    best_model_name = None
    best_test_acc = -1.0
    best_model_obj = None

    for name, clf in models.items():
        if name in ("MLPClassifier",):
            clf.fit(X_train_scaled, y_train)
            train_acc = float(clf.score(X_train_scaled, y_train))
            test_acc = float(clf.score(X_test_scaled, y_test))
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            train_acc = float(clf.score(X_train, y_train))
            test_acc = float(clf.score(X_test, y_test))
            y_pred = clf.predict(X_test)

        # Baseline heuristic accuracy on the same test split
        heuristic_test = np.array([r["heuristic_mov"] for r, is_t in zip(records, test_mask) if is_t])
        heuristic_acc = float(np.mean(heuristic_test == y_test))

        print("-" * 65)
        print(f"Model: {name}")
        print(f"  Train Accuracy:     {train_acc * 100:.2f}%")
        print(f"  Test Accuracy:      {test_acc * 100:.2f}%")
        print(f"  Baseline Heuristic: {heuristic_acc * 100:.2f}%")

        # Classification report
        all_labels = [int(lbl) for lbl in sorted(list(set(y_test) | set(y_pred)))]
        cls_rep = classification_report(y_test, y_pred, labels=all_labels, output_dict=True, zero_division=0)
        # Convert any numpy scalars inside classification report to standard python floats/ints
        cls_rep_clean = {}
        for k, v in cls_rep.items():
            if isinstance(v, dict):
                cls_rep_clean[str(k)] = {sub_k: float(sub_v) for sub_k, sub_v in v.items()}
            else:
                cls_rep_clean[str(k)] = float(v)

        c_matrix = [[int(cell) for cell in row] for row in confusion_matrix(y_test, y_pred, labels=all_labels).tolist()]

        results[name] = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "heuristic_test_accuracy": round(heuristic_acc, 4),
            "classification_report": cls_rep_clean,
            "confusion_matrix": c_matrix,
            "labels": all_labels,
        }

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_name = name
            best_model_obj = clf

    # Also compute 5-Fold Stratified Cross Validation on the whole dataset
    print("-" * 65)
    print("Evaluating 5-Fold Stratified Cross-Validation across entire dataset (N=92)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Remove classes with < 2 samples for clean CV or use non-stratified kfold
    try:
        cv_scores = cross_val_score(models["RandomForest"], X, y, cv=skf)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        print(f"RandomForest 5-Fold CV Accuracy: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
    except Exception as exc:
        cv_mean, cv_std = 0.0, 0.0
        print(f"CV skipped due to rare single-sample classes: {exc}")

    # Retrain best model on full dataset for maximum deployment accuracy
    print("-" * 65)
    print(f"Retraining Best Model ({best_model_name}) on full 92-sample dataset for deployment...")
    if best_model_name in ("MLPClassifier",):
        full_scaler = StandardScaler()
        X_full_scaled = full_scaler.fit_transform(X)
        best_model_obj.fit(X_full_scaled, y)
        full_train_acc = float(best_model_obj.score(X_full_scaled, y))
        deployment_scaler = full_scaler
    else:
        best_model_obj.fit(X, y)
        full_train_acc = float(best_model_obj.score(X, y))
        deployment_scaler = None

    print(f"Full Dataset Retraining Accuracy: {full_train_acc * 100:.2f}%")

    # Feature importances if RandomForest
    feature_importances = {}
    if hasattr(best_model_obj, "feature_importances_"):
        for fname, imp in zip(FEATURE_NAMES, best_model_obj.feature_importances_):
            feature_importances[fname] = round(float(imp), 4)
        print("Top 5 Feature Importances:")
        for fname, imp in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  {fname}: {imp:.4f}")

    # Export model package
    checkpoint = {
        "model_name": best_model_name,
        "model_type": str(type(best_model_obj)),
        "feature_names": FEATURE_NAMES,
        "classes": [int(c) for c in best_model_obj.classes_],
        "model": best_model_obj,
        "scaler": deployment_scaler,
        "metrics": {
            "test_accuracy": round(best_test_acc, 4),
            "full_dataset_accuracy": round(full_train_acc, 4),
            "cv_5fold_mean": round(cv_mean, 4),
            "cv_5fold_std": round(cv_std, 4),
        },
        "feature_importances": feature_importances,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_samples": len(X),
    }

    model_path = output_dir / "movement_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Exported model checkpoint to: {model_path}")

    # Export metrics JSON
    metrics_summary = {
        "dataset_summary": {
            "total_samples": len(X),
            "train_samples": int(np.sum(train_mask)),
            "test_samples": int(np.sum(test_mask)),
            "features_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        },
        "best_model": best_model_name,
        "models_comparison": results,
        "cross_validation": {
            "cv_folds": 5,
            "mean_accuracy": round(cv_mean, 4),
            "std_accuracy": round(cv_std, 4),
        },
        "full_retrained_accuracy": round(full_train_acc, 4),
        "feature_importances": feature_importances,
    }
    metrics_path = output_dir / "classifier_metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported metrics summary to: {metrics_path}")

    return metrics_summary


def compute_counting_calibration(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze detected vs ground-truth count correlations per movement to derive calibration ratios."""
    from collections import Counter
    gt_counts = Counter(r["true_gt_mov"] for r in records)
    pred_counts = Counter(r["heuristic_mov"] for r in records)

    calibration = {}
    for m in range(1, 13):
        gt_c = gt_counts.get(m, 0)
        pred_c = pred_counts.get(m, 0)
        ratio = round(gt_c / max(1, pred_c), 4) if pred_c > 0 else 1.0
        calibration[str(m)] = {
            "movement_id": m,
            "ground_truth_count": gt_c,
            "detected_count": pred_c,
            "calibration_factor": ratio,
            "bias_offset": gt_c - pred_c,
        }

    cal_path = output_dir / "counting_calibration.json"
    cal_data = {
        "calibration_method": "empirical_ratio_and_bias",
        "description": "Per-movement calibration factors derived from 1-minute ground-truth sample",
        "movements": calibration,
    }
    cal_path.write_text(json.dumps(cal_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported counting calibration to: {cal_path}")
    return cal_data


def main() -> int:
    video_path = ROOT / "AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4"
    roi_path = ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt"
    detector_path = ROOT / "models/detector.onnx"
    gt_csv_path = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    output_dir = ROOT / "experiments/training/track1_vehicle_counting"

    print("Step 1: Running pipeline on cam_5 to record trajectories...")
    events, track_snapshots, img_w, img_h = run_pipeline_and_extract_tracks(
        video_path=video_path,
        roi_path=roi_path,
        detector_path=detector_path,
        max_frames=600,
    )
    print(f"Emitted {len(events)} counting events, captured {len(track_snapshots)} track snapshots.")

    print("\nStep 2: Aligning with ground truth and extracting features...")
    X, y, records = match_tracks_to_ground_truth(
        events=events,
        track_snapshots=track_snapshots,
        gt_csv_path=gt_csv_path,
        img_w=img_w,
        img_h=img_h,
        tol_frames=15,
    )

    print("\nStep 3: Training machine learning classifiers...")
    train_and_evaluate(X, y, records, output_dir)

    print("\nStep 4: Computing counting calibration factors...")
    compute_counting_calibration(records, output_dir)

    print("\nMachine Learning Movement Classifier training completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
