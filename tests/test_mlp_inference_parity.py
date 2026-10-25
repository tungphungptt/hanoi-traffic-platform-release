"""Regression tests for MLP movement classifier inference parity, scaler compatibility, and fallback."""
from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import pytest

from aicc.track1_counting import (
    Track1MovementMatcher,
    Track1VehicleTracker,
    Track1Detection,
    extract_trajectory_features,
    TRAJECTORY_FEATURE_NAMES,
    CAM5_MOVEMENTS,
)
from tools.evaluate_heldout_track1 import evaluate_heldout_slice, load_csv_rows

ROOT = Path(__file__).resolve().parents[1]


def _resolve_mlp_checkpoint() -> Path:
    cand_pkg = ROOT / "models/movement_classifier_train_only.pkl"
    if cand_pkg.is_file():
        return cand_pkg
    cand_art = ROOT / "artifacts/track1_final/models/movement_classifier_train_only.pkl"
    if cand_art.is_file():
        return cand_art
    return cand_pkg


def test_feature_extraction_schema_and_ranges():
    """Verify trajectory feature extractor produces standard 15-dim schema and physical ranges."""
    det_start = Track1Detection(box=(500.0, 400.0, 40.0, 40.0), score=0.9, coco_class_id=2, track1_class_id=1, center=(500.0, 400.0))
    track = Track1VehicleTracker(track_id=1, initial_det=det_start, initial_frame=100, in_roi=True)

    # Simulate a 50-frame vehicle moving across the scene
    for f in range(101, 150):
        det = Track1Detection(box=(float(f * 5 - 20), float(f * 4 - 20), 40.0, 40.0), score=0.9, coco_class_id=2, track1_class_id=1, center=(float(f * 5), float(f * 4)))
        track.update(det, frame_idx=f, in_roi=True)

    feats = extract_trajectory_features(track, img_w=1280.0, img_h=960.0)

    # 1. Dimension and names parity
    assert len(feats) == 15
    assert len(TRAJECTORY_FEATURE_NAMES) == 15

    # 2. Check feature 9 is real video frame duration (not capped sequence length <= 20)
    expected_duration = 149 - 100  # 49 frames
    assert feats[9] == pytest.approx(expected_duration, rel=1e-3)
    assert feats[9] > 20.0, "Feature 9 must represent true frame duration, not subsampled L <= 20"

    # 3. Check normalized travel distance feature 6
    dx = 149 * 5 - 500.0
    dy = 149 * 4 - 400.0
    expected_dist_norm = np.hypot(dx, dy) / np.hypot(1280.0, 960.0)
    assert feats[6] == pytest.approx(expected_dist_norm, abs=1e-4)


def test_mlp_checkpoint_scaler_compatibility():
    """Verify that the deployed MLP checkpoint scaler matches video duration distributions."""
    ckpt_path = _resolve_mlp_checkpoint()
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    scaler = ckpt.get("scaler")
    model = ckpt.get("model")
    assert scaler is not None, "MLP checkpoint must include a fitted StandardScaler"
    assert model is not None, "MLP checkpoint must include a trained model"

    # In a video tracklet dataset, mean duration is ~60-80 frames, not ~20 frames
    mean_duration = scaler.mean_[9]
    scale_duration = scaler.scale_[9]

    assert mean_duration > 35.0, (
        f"Scaler mean duration is {mean_duration:.2f}, indicating subsampled sequence length (L <= 20) "
        "rather than true frame duration!"
    )
    assert scale_duration > 10.0, f"Scaler scale duration is suspiciously narrow: {scale_duration:.2f}"

    # Verify classes
    classes = [int(c) for c in model.classes_]
    assert 5 in classes, "Movement 5 must be in trained model classes"
    assert 3 in classes, "Movement 3 must be in trained model classes"
    assert 11 in classes, "Movement 11 must be in trained model classes"


def test_mlp_matcher_fallback_on_empty_and_short_tracks():
    """Verify graceful fallback to geometric heuristic on degenerate, missing or short tracks."""
    ckpt_path = _resolve_mlp_checkpoint()
    matcher = Track1MovementMatcher(CAM5_MOVEMENTS, ml_model_path=ckpt_path)

    # 1. 1-point track
    det_1pt = Track1Detection(box=(480.0, 480.0, 40.0, 40.0), score=0.9, coco_class_id=2, track1_class_id=1, center=(500.0, 500.0))
    track_1pt = Track1VehicleTracker(track_id=2, initial_det=det_1pt, initial_frame=10, in_roi=True)
    mov_1pt = matcher.match(track_1pt)
    assert 1 <= mov_1pt <= 12


def test_mlp_confidence_fallback_on_unseen_or_ambiguous_motion():
    """Verify that ambiguous or out-of-distribution tracks fall back to heuristic rather than predicting wrong."""
    ckpt_path = _resolve_mlp_checkpoint()
    matcher = Track1MovementMatcher(CAM5_MOVEMENTS, ml_model_path=ckpt_path)

    # Create a track that moves in an unnatural circular or stationary path
    det_ambig = Track1Detection(box=(380.0, 380.0, 40.0, 40.0), score=0.9, coco_class_id=2, track1_class_id=1, center=(400.0, 400.0))
    ambig_track = Track1VehicleTracker(track_id=3, initial_det=det_ambig, initial_frame=0, in_roi=True)
    for i in range(1, 30):
        # Stationary jitter around (400, 400)
        det = Track1Detection(box=(380.0, 380.0, 40.0, 40.0), score=0.9, coco_class_id=2, track1_class_id=1, center=(400.0 + np.sin(i) * 2, 400.0 + np.cos(i) * 2))
        ambig_track.update(det, frame_idx=i, in_roi=True)

    # The matcher should return a valid movement ID without crashing
    mov_ambig = matcher.match(ambig_track)
    assert 1 <= mov_ambig <= 12


def test_revalidated_mlp_csv_heldout_parity():
    """Historical regression snapshot fixture on development interval (frames 351–600).
    
    NOTE: This test serves as an engineering quality gate / regression snapshot to prevent
    silent feature/scaler breakage (such as the historical F1 = 0.16 failure). It does NOT
    constitute proof of independent cross-camera generalization, because geometric ROI
    boundaries were previously inspected and tuned on this clip during development.
    Do not tune models further simply to exceed this benchmark.
    """
    reval_csv = ROOT / "artifacts/deployment_validation/mlp_cam5_600frames_revalidated.csv"
    gt_csv = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"

    if not reval_csv.is_file() or not gt_csv.is_file():
        pytest.skip("Test requires AI City Challenge dataset and deployment artifacts (excluded from portable zip)")

    gt_rows = load_csv_rows(gt_csv)
    reval_rows = load_csv_rows(reval_csv)

    res = evaluate_heldout_slice(reval_rows, gt_rows, min_frame=351, max_frame=600, tolerance_frames=15)

    # Invariants
    assert res["total_ground_truth"] == 47
    assert res["total_predicted"] == 53

    # Count-only evaluation invariant
    assert res["count_only_matching"]["tp"] == 46
    assert res["count_only_matching"]["fp"] == 7
    assert res["count_only_matching"]["fn"] == 1
    assert res["count_only_matching"]["f1_score"] == 0.92

    # Strict movement matching must be high (F1 >= 0.80, TP >= 40), resolving the F1 = 0.16 regression
    assert res["strict_matching"]["tp"] >= 40, f"Expected TP >= 40, got {res['strict_matching']['tp']}"
    assert res["strict_matching"]["f1_score"] >= 0.80, f"Expected F1 >= 0.80, got {res['strict_matching']['f1_score']}"
    assert res["strict_matching"]["wrong_movement_count"] <= 6
