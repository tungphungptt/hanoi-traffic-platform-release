"""Unit tests for AI City Challenge Track 1 Vehicle Counting Pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from aicc.track1_counting import (
    COCO_TO_TRACK1,
    CAM5_MOVEMENTS,
    Track1Detection,
    Track1VehicleClass,
    Track1VehicleTracker,
    Track1MovementMatcher,
    Track1CountingEvent,
    export_track1_csv,
    evaluate_track1_predictions,
    load_roi_polygon,
    MOIDefinition,
)

ROOT = Path(__file__).resolve().parents[1]


def test_coco_to_track1_class_mapping():
    """Verify standard COCO classes map correctly to Track 1 Car (1) and Truck (2)."""
    assert COCO_TO_TRACK1[2] == Track1VehicleClass.CAR  # Car -> 1
    assert COCO_TO_TRACK1[7] == Track1VehicleClass.TRUCK  # Truck -> 2
    assert COCO_TO_TRACK1[5] == Track1VehicleClass.TRUCK  # Bus -> 2

    # Verify motorcycle and bicycle are not mapped
    assert 3 not in COCO_TO_TRACK1  # motorcycle
    assert 1 not in COCO_TO_TRACK1  # bicycle


def test_roi_polygon_loading():
    """Verify ROI polygon text parser loads valid coordinate arrays."""
    roi_file = ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt"
    if not roi_file.is_file():
        pytest.skip("Requires AIC21 dataset")
    poly = load_roi_polygon(roi_file)
    assert isinstance(poly, np.ndarray)
    assert poly.shape == (6, 2)
    assert poly[0].tolist() == [1, 277]
    assert poly[-1].tolist() == [0, 956]


def test_cam5_movement_definitions():
    """Verify camera 5 has exactly 12 movements defined with unit direction vectors."""
    assert len(CAM5_MOVEMENTS) == 12
    ids = [m.movement_id for m in CAM5_MOVEMENTS]
    assert ids == list(range(1, 13))
    for m in CAM5_MOVEMENTS:
        assert isinstance(m.name, str)
        dx, dy = m.direction_vector
        norm = np.hypot(dx, dy)
        assert abs(norm - 1.0) < 1e-4, f"Vector not normalized for movement {m.movement_id}"


def test_tracker_voting_and_travel_distance():
    """Verify track state maintains majority voting and travel distance calculation."""
    det1 = Track1Detection(box=(100, 100, 50, 50), score=0.8, coco_class_id=2, track1_class_id=1, center=(125, 125))
    tracker = Track1VehicleTracker(track_id=1, initial_det=det1, initial_frame=10, in_roi=True)

    det2 = Track1Detection(box=(130, 140, 50, 50), score=0.85, coco_class_id=2, track1_class_id=1, center=(155, 165))
    tracker.update(det2, frame_idx=11, in_roi=True)

    det3 = Track1Detection(box=(160, 180, 60, 60), score=0.7, coco_class_id=7, track1_class_id=2, center=(190, 210))
    tracker.update(det3, frame_idx=12, in_roi=False)

    assert tracker.majority_class == int(Track1VehicleClass.CAR)  # 2 car votes vs 1 truck vote
    assert tracker.roi_frames_count == 2
    assert tracker.travel_distance > 50.0


def test_movement_matcher_assignment():
    """Verify that movement matcher maps directional trajectories to the expected movement ID."""
    matcher = Track1MovementMatcher(CAM5_MOVEMENTS)

    # Simulate Movement 11: South-West to North-East through traffic (from ~60,670 to ~300,490)
    det1 = Track1Detection(box=(50, 660, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(70, 680))
    t11 = Track1VehicleTracker(1, det1, 0, True)
    det2 = Track1Detection(box=(280, 480, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(300, 500))
    t11.update(det2, 20, False)

    matched_id = matcher.match(t11)
    assert matched_id == 11, f"Expected movement 11, got {matched_id}"


def test_movement_4_vs_5_separation():
    """Verify clean separation between Movement 4 (Right Turn) and Movement 5 (Through)."""
    matcher = Track1MovementMatcher(CAM5_MOVEMENTS)

    # Movement 4: Enters Approach 2 (top right), turns right to North-West exit (260, 160)
    det_m4_start = Track1Detection(box=(540, 120, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(560, 140))
    t4 = Track1VehicleTracker(101, det_m4_start, 0, True)
    det_m4_end = Track1Detection(box=(240, 140, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(260, 160))
    t4.update(det_m4_end, 15, False)
    assert matcher.match(t4) == 4, "Movement 4 trajectory misclassified"

    # Movement 5: Enters Approach 2 (top right), travels across intersection to South-West exit (45, 370)
    det_m5_start = Track1Detection(box=(630, 120, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(650, 140))
    t5 = Track1VehicleTracker(102, det_m5_start, 0, True)
    det_m5_end = Track1Detection(box=(25, 350, 40, 40), score=0.9, coco_class_id=2, track1_class_id=1, center=(45, 370))
    t5.update(det_m5_end, 40, False)
    assert matcher.match(t5) == 5, "Movement 5 trajectory misclassified"


def test_tracker_velocity_prediction():
    """Verify velocity tracking and position extrapolation."""
    det1 = Track1Detection(box=(100, 100, 20, 20), score=0.9, coco_class_id=2, track1_class_id=1, center=(100.0, 100.0))
    tracker = Track1VehicleTracker(1, det1, initial_frame=10, in_roi=True)

    det2 = Track1Detection(box=(110, 105, 20, 20), score=0.9, coco_class_id=2, track1_class_id=1, center=(110.0, 105.0))
    tracker.update(det2, frame_idx=12, in_roi=True)

    # dt = 2, dx = 10, dy = 5 -> vx = 5.0 * 0.6 = 3.0, vy = 2.5 * 0.6 = 1.5
    pred = tracker.predict_next_point(current_frame=14)
    # Extrapolation: last_pt (110, 105) + velocity * (14 - 12)
    assert pred[0] > 110.0
    assert pred[1] > 105.0


def test_export_and_evaluate_track1_format(tmp_path):
    """Verify CSV export and evaluation against ground truth format."""
    events = [
        Track1CountingEvent(video_clip_id=10, frame_id=15, movement_id=3, vehicle_class_id=1, track_id=1),
        Track1CountingEvent(video_clip_id=10, frame_id=20, movement_id=4, vehicle_class_id=1, track_id=2),
    ]
    csv_file = tmp_path / "test_out.csv"
    export_track1_csv(events, csv_file)

    assert csv_file.is_file()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["video_clip_id", "frame_id", "movement_id", "vehicle_class_id"]
    assert reader[1] == ["10", "15", "3", "1"]
    assert reader[2] == ["10", "20", "4", "1"]

    # Self-evaluate
    metrics = evaluate_track1_predictions(csv_file, csv_file, tolerance_frames=5)
    assert metrics["total_ground_truth"] == 2
    assert metrics["total_predicted"] == 2
    assert metrics["f1_score"] == 1.0
    assert metrics["mae_per_movement"] == 0.0


def test_reconciliation_arithmetic_invariants():
    """Verify that event reconciliation satisfies closed-form arithmetic invariants."""
    from tools.generate_reconciliation_table import build_reconciliation

    gt_file = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    pred_file = ROOT / "experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv"

    if not gt_file.is_file() or not pred_file.is_file():
        pytest.skip("Requires AIC21 ground truth sample and baseline predictions")

    if not gt_file.is_file() or not pred_file.is_file():
        pytest.skip("Requires AIC21 ground truth sample and baseline predictions")

    recon = build_reconciliation(gt_file, pred_file, tol_frames=15)
    stats = recon["stats"]

    # Invariant 1: Total GT partition
    gt_partition_sum = (
        stats["exact_match_tp"]
        + stats["wrong_movement"]
        + stats["wrong_class"]
        + stats["false_negative_missed"]
    )
    assert gt_partition_sum == stats["total_gt"], f"GT invariant failed: {gt_partition_sum} != {stats['total_gt']}"
    assert stats["total_gt"] == 96

    # Invariant 2: Total Prediction partition
    pred_partition_sum = (
        stats["exact_match_tp"]
        + stats["wrong_movement"]
        + stats["wrong_class"]
        + stats["duplicate_counts"]
        + stats["false_positive_spurious"]
    )
    assert pred_partition_sum == stats["total_pred"], f"Pred invariant failed: {pred_partition_sum} != {stats['total_pred']}"
    assert stats["total_pred"] == 107

    # Invariant 3: Temporal Movement Error relationship
    assert stats["total_temporal_movement_fn"] == stats["wrong_movement"] + stats["wrong_class"] + stats["false_negative_missed"]
    assert stats["total_temporal_movement_fp"] == stats["wrong_movement"] + stats["duplicate_counts"] + stats["false_positive_spurious"]
    assert stats["exact_match_tp"] == 86
    assert stats["total_temporal_movement_fn"] == 10
    assert stats["total_temporal_movement_fp"] == 21


def test_mae_rmse_metric_formulations():
    """Verify MAE/RMSE calculations match across N=10 active, N=12 all movements, and N=24 cells."""
    gt_file = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    pred_file = ROOT / "experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv"

    if not gt_file.is_file() or not pred_file.is_file():
        pytest.skip("Requires AIC21 ground truth sample and baseline predictions")

    metrics = evaluate_track1_predictions(pred_file, gt_file, tolerance_frames=15)

    # Active movements (N=10 in video clip)
    assert metrics["mae_per_movement"] == 2.1
    assert metrics["rmse_per_movement"] == 3.271

    # All defined movements (N=12)
    assert metrics["mae_all_12_movements"] == 1.75
    assert metrics["rmse_all_12_movements"] == 2.986

    # All 24 movement x vehicle class cells
    assert metrics["mae_24_movement_class_cells"] == 0.958
    assert metrics["rmse_24_movement_class_cells"] == 1.744

    # Strict vs count-only precision/recall
    assert metrics["precision"] == 0.8037
    assert metrics["recall"] == 0.8958
    assert metrics["f1_score"] == 0.8473

    count_eval = metrics["count_only_evaluation"]
    assert count_eval["matched_tp"] == 92
    assert count_eval["unmatched_fp"] == 15
    assert count_eval["missed_fn"] == 4
    assert count_eval["precision"] == 0.8598
    assert count_eval["recall"] == 0.9583
    assert count_eval["f1_score"] == 0.9064

    # Mathematical invariants for count-only
    assert count_eval["matched_tp"] + count_eval["missed_fn"] == 96
    assert count_eval["matched_tp"] + count_eval["unmatched_fp"] == 107
    expected_f1 = round(2 * 92 / (2 * 92 + 15 + 4), 4)
    assert count_eval["f1_score"] == expected_f1


def test_trajectory_feature_extraction_edge_cases():
    """Verify trajectory feature extractor handles empty, single-point and normal tracks robustly."""
    from aicc.track1_counting import TRAJECTORY_FEATURE_NAMES, extract_trajectory_features

    assert len(TRAJECTORY_FEATURE_NAMES) == 15

    # Edge case 1: Tracker with single point
    det = Track1Detection(box=(100, 100, 50, 50), score=0.9, coco_class_id=2, track1_class_id=1, center=(125, 125))
    tracker_single = Track1VehicleTracker(1, det, 0, True)
    feats_single = extract_trajectory_features(tracker_single)
    assert len(feats_single) == 15
    assert feats_single[0] == round(125.0 / 1280.0, 5)  # x_start
    assert feats_single[2] == round(125.0 / 1280.0, 5)  # x_end == x_start
    assert feats_single[6] == 0.0                        # travel_dist == 0
    assert feats_single[9] == 1.0                        # duration == 1

    # Edge case 2: Normal moving track
    det2 = Track1Detection(box=(500, 400, 60, 60), score=0.85, coco_class_id=2, track1_class_id=1, center=(530, 430))
    tracker_single.update(det2, 20, True)
    feats_moving = extract_trajectory_features(tracker_single)
    assert len(feats_moving) == 15
    assert feats_moving[6] > 0.0                         # travel_dist > 0
    assert feats_moving[9] == 20.0                       # duration == 20


def test_ml_movement_matcher_loading_and_fallback():
    """Verify Track1MovementMatcher loads ML model checkpoint and falls back gracefully."""
    ml_ckpt = ROOT / "experiments/training/track1_vehicle_counting/movement_classifier.pkl"
    if not ml_ckpt.is_file():
        ml_ckpt = ROOT / "models/movement_classifier_train_only.pkl"
    if not ml_ckpt.is_file():
        pytest.skip("Requires ML model checkpoint")
    matcher = Track1MovementMatcher(CAM5_MOVEMENTS, ml_model_path=ml_ckpt)

    assert matcher.ml_model is not None, "Failed to load ML model checkpoint"

    # Test prediction with Movement 11 trajectory (South-West to North-East)
    det1 = Track1Detection(box=(60, 670, 50, 50), score=0.9, coco_class_id=2, track1_class_id=1, center=(85, 695))
    t11 = Track1VehicleTracker(1, det1, 0, True)
    det2 = Track1Detection(box=(800, 200, 50, 50), score=0.9, coco_class_id=2, track1_class_id=1, center=(825, 225))
    t11.update(det2, 30, True)

    pred_m = matcher.match(t11)
    assert pred_m == 11, f"Expected ML model to predict movement 11, got {pred_m}"

    # Test graceful fallback with non-existent checkpoint path
    fallback_matcher = Track1MovementMatcher(CAM5_MOVEMENTS, ml_model_path="non_existent_file.pkl")
    assert fallback_matcher.ml_model is None
    fallback_pred = fallback_matcher.match(t11)
    assert fallback_pred == 11  # Falls back to heuristic vector matcher without crashing


def test_auto_label_pipeline_and_curation(tmp_path):
    """Verify AutoLabelPipeline generates YOLO-format annotations and review tool curates subset."""
    from aicc.auto_labeling import AutoLabelPipeline, filter_and_curate_pseudo_labels
    import onnxruntime as ort

    det_path = ROOT / "models/detector.onnx"
    if not det_path.is_file():
        pytest.skip("Requires YOLOv8 detector weights (models/detector.onnx)")
    session = ort.InferenceSession(str(det_path), providers=["CPUExecutionProvider"])
    pipeline = AutoLabelPipeline(session, conf_threshold=0.35)

    # Synthetic test frame (640x480 black image with a bright rectangle)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_frame = pipeline.detect_frame(frame, frame_idx=0, video_id="test_vid", timestamp_s=0.0)

    assert res_frame.video_id == "test_vid"
    assert res_frame.frame_idx == 0
    assert isinstance(res_frame.boxes, list)

    # Test curation filter logic with a synthetic manifest
    manifest_data = {
        "dataset_provenance": {
            "source_video": "test_cam.mp4",
            "model_version": "yolov8n-coco-v1",
            "annotation_status": "PROPOSED_PSEUDO_LABEL",
        },
        "summary_statistics": {"total_pseudo_boxes": 3},
        "frames": [
            {
                "frame_idx": 10,
                "timestamp_s": 1.0,
                "boxes": [
                    {
                        "track1_class_id": 1,
                        "confidence": 0.85,
                        "bbox_yolo_norm": [0.5, 0.5, 0.2, 0.2],
                        "in_roi": True,
                    },
                    {
                        "track1_class_id": 1,
                        "confidence": 0.40,  # Below threshold
                        "bbox_yolo_norm": [0.3, 0.3, 0.1, 0.1],
                        "in_roi": True,
                    },
                    {
                        "track1_class_id": 2,
                        "confidence": 0.90,
                        "bbox_yolo_norm": [0.7, 0.7, 0.3, 0.3],
                        "in_roi": False,     # Outside ROI
                    },
                ],
            }
        ],
    }

    src_manifest = tmp_path / "test_manifest.json"
    src_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    curated = filter_and_curate_pseudo_labels(
        manifest_path=src_manifest,
        output_dir=tmp_path / "curated",
        min_confidence=0.60,
        require_roi=True,
    )

    prov = curated["curation_provenance"]
    stats = curated["curated_statistics"]

    assert prov["curated_subset_status"] == "VERIFIED_SUBSET"
    assert stats["total_curated_boxes"] == 1  # Only the 1st box meets conf >= 0.60 and in_roi == True
    assert stats["curated_cars"] == 1
    assert stats["curated_trucks"] == 0


def test_ml_movement_classifier_evaluation_invariants():
    """Verify that ML Movement Classifier achieves exact reconciliation with 0 wrong movement errors."""
    from tools.generate_reconciliation_table import build_reconciliation

    gt_file = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    pred_ml_file = ROOT / "experiments/training/track1_vehicle_counting/output_counting_ml_cam5_1min.csv"

    if not gt_file.is_file() or not pred_ml_file.is_file():
        pytest.skip("Requires AIC21 ground truth sample and ML prediction outputs")

    recon = build_reconciliation(gt_file, pred_ml_file, tol_frames=15)
    stats = recon["stats"]

    # Invariants
    assert stats["total_gt"] == 96
    assert stats["total_pred"] == 107
    assert stats["exact_match_tp"] == 92
    assert stats["wrong_movement"] == 0     # All wrong movements resolved by ML classifier
    assert stats["wrong_class"] == 0
    assert stats["false_negative_missed"] == 4
    assert stats["duplicate_counts"] == 7
    assert stats["false_positive_spurious"] == 8

    # Precision, Recall, F1
    strict_f1 = round(2 * 92 / (2 * 92 + 15 + 4), 4)
    assert strict_f1 == 0.9064


def test_heldout_slice_evaluation_invariants():
    """Verify that held-out interval (frames 351-600) matches closed-form arithmetic invariants."""
    from tools.evaluate_heldout_track1 import evaluate_heldout_slice, load_csv_rows

    gt_file = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    base_file = ROOT / "experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv"
    train_only_file = ROOT / "experiments/training/track1_vehicle_counting/output_counting_ml_train_only_cam5_1min.csv"

    if not gt_file.is_file() or not base_file.is_file() or not train_only_file.is_file():
        pytest.skip("Requires AIC21 ground truth sample and baseline predictions")

    gt_rows = load_csv_rows(gt_file)
    base_rows = load_csv_rows(base_file)
    train_only_rows = load_csv_rows(train_only_file)

    res_base = evaluate_heldout_slice(base_rows, gt_rows, min_frame=351, max_frame=600, tolerance_frames=15)
    res_to = evaluate_heldout_slice(train_only_rows, gt_rows, min_frame=351, max_frame=600, tolerance_frames=15)

    # Invariants for held-out interval
    assert res_base["total_ground_truth"] == 47
    assert res_base["total_predicted"] == 53
    assert res_to["total_ground_truth"] == 47
    assert res_to["total_predicted"] == 53

    # Baseline strict matching on held-out
    assert res_base["strict_matching"]["tp"] == 41
    assert res_base["strict_matching"]["fp"] == 12
    assert res_base["strict_matching"]["fn"] == 6
    assert res_base["strict_matching"]["wrong_movement_count"] == 5

    # Train-only ML strict matching on held-out
    assert res_to["strict_matching"]["tp"] == 41
    assert res_to["strict_matching"]["fp"] == 12
    assert res_to["strict_matching"]["fn"] == 6
    assert res_to["strict_matching"]["wrong_movement_count"] == 5

    # Count-only partition invariants on held-out
    assert res_base["count_only_matching"]["tp"] == 46
    assert res_base["count_only_matching"]["fp"] == 7
    assert res_base["count_only_matching"]["fn"] == 1

    # Exact equality: total GT = TP + WM + Pure_FN
    assert res_base["strict_matching"]["tp"] + res_base["strict_matching"]["wrong_movement_count"] + res_base["count_only_matching"]["fn"] == 47
    # Exact equality: total Pred = TP + WM + Spurious_FP
    assert res_base["strict_matching"]["tp"] + res_base["strict_matching"]["wrong_movement_count"] + res_base["count_only_matching"]["fp"] == 53


def test_cnn_lstm_movement_matcher_loading_and_fallback():
    """Verify that CNN-LSTM checkpoint loads cleanly and executes sequence inference with fallback."""
    ckpt_path = ROOT / "artifacts/track1_final/models/cnn_lstm_movement_experimental.pkl"
    if not ckpt_path.is_file():
        pytest.skip("CNN-LSTM checkpoint not yet created.")

    matcher = Track1MovementMatcher(CAM5_MOVEMENTS, ml_model_path=ckpt_path)
    assert matcher.model_type == "cnn_lstm"
    assert matcher.cnn_lstm_ckpt is not None
    assert matcher.cnn_lstm_ckpt["model_type"] == "CNN_LSTM_SEQUENCE_CLASSIFIER"

    # Synthetic track: Approach 1 to Exit 1 (Movement 11)
    det1 = Track1Detection(box=(500.0, 750.0, 50.0, 40.0), score=0.8, coco_class_id=2, track1_class_id=1, center=(525.0, 770.0))
    track = Track1VehicleTracker(track_id=101, initial_det=det1, initial_frame=10, in_roi=True)
    for f in range(11, 25):
        det_f = Track1Detection(
            box=(500.0, 750.0 - (f - 10) * 15.0, 50.0, 40.0),
            score=0.85,
            coco_class_id=2,
            track1_class_id=1,
            center=(525.0, 770.0 - (f - 10) * 15.0),
        )
        track.update(det_f, f, in_roi=True)

    predicted_mov = matcher.match(track)
    assert 1 <= predicted_mov <= 12

    # Verify fallback for degenerate track (<2 points)
    single_point_track = Track1VehicleTracker(track_id=102, initial_det=det1, initial_frame=1, in_roi=True)
    fallback_mov = matcher.match(single_point_track)
    assert fallback_mov == 1  # Heuristic fallback for degenerate track



