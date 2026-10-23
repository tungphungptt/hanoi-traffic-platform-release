#!/usr/bin/env python3
"""CLI runner for AI City Challenge Track 1 Vehicle Counting.

Executes baseline detection -> tracking -> ROI -> MOI counting -> evaluation on Dataset_A.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import onnxruntime as ort

from aicc.track1_counting import (
    CAM5_MOVEMENTS,
    Track1CountingPipeline,
    export_track1_csv,
    evaluate_track1_predictions,
    load_roi_polygon,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt",
        help="Path to ROI polygon text file.",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=ROOT / "models/detector.onnx",
        help="Path to detector ONNX model.",
    )
    parser.add_argument(
        "--video-clip-id",
        type=int,
        default=10,
        help="Video clip ID to write in the output CSV (10 for cam_5 in sample ground truth).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum frames to process (600 frames = 1 minute at 10 FPS; 0 or negative for all).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv",
        help="Destination path for the Track 1 CSV counting output.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv",
        help="Optional path to reference ground truth CSV for evaluation.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "experiments/training/track1_vehicle_counting/evaluation_metrics.json",
        help="Destination path for evaluation metrics JSON.",
    )
    parser.add_argument(
        "--movement-classifier-type",
        type=str,
        choices=["heuristic", "mlp", "cnn_lstm"],
        default="heuristic",
        help="Movement classification approach (production default: heuristic; experimental: mlp, cnn_lstm).",
    )
    parser.add_argument(
        "--ml-movement-classifier",
        type=Path,
        default=None,
        help="Optional explicit path to trained movement classifier checkpoint (.pkl).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.is_file():
        print(f"[ERROR] Video file not found: {args.video}", file=sys.stderr)
        return 1
    if not args.roi.is_file():
        print(f"[ERROR] ROI file not found: {args.roi}", file=sys.stderr)
        return 1
    if not args.detector.is_file():
        print(f"[ERROR] Detector model not found: {args.detector}", file=sys.stderr)
        return 1

    # Resolve model path based on classifier type
    ml_model_path = args.ml_movement_classifier
    if ml_model_path is None:
        if args.movement_classifier_type == "cnn_lstm":
            cand = ROOT / "artifacts/track1_final/models/cnn_lstm_movement_experimental.pkl"
            if cand.is_file():
                ml_model_path = cand
        elif args.movement_classifier_type == "mlp":
            cand = ROOT / "artifacts/track1_final/models/movement_classifier_train_only.pkl"
            if cand.is_file():
                ml_model_path = cand

    print("=" * 60)
    print("AI City Challenge Track 1 - Vehicle Counting Pipeline")
    print("=" * 60)
    print(f"Video file:      {args.video}")
    print(f"ROI polygon:     {args.roi}")
    print(f"Detector ONNX:   {args.detector}")
    print(f"Video clip ID:   {args.video_clip_id}")
    print(f"Classifier Type: {args.movement_classifier_type} (Model: {ml_model_path})")

    # Load ROI polygon
    roi_poly = load_roi_polygon(args.roi)
    print(f"ROI vertices:    {len(roi_poly)} points")

    # Load ONNX detector session
    session = ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])

    # Configure pipeline
    movements = CAM5_MOVEMENTS
    pipeline = Track1CountingPipeline(
        detector_session=session,
        roi_polygon=roi_poly,
        movements=movements,
        video_clip_id=args.video_clip_id,
        conf_threshold=args.conf_threshold,
        ml_model_path=ml_model_path,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}", file=sys.stderr)
        return 1

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    limit_frames = args.max_frames if args.max_frames > 0 else total_video_frames
    frames_to_read = min(limit_frames, total_video_frames)
    print(f"Video FPS:       {fps:.2f}, Total frames: {total_video_frames}, Target frames: {frames_to_read}")

    start_time = time.time()
    frame_idx = 0
    last_log = 0

    try:
        while frame_idx < frames_to_read:
            ok, frame = cap.read()
            if not ok:
                break
            pipeline.process_frame(frame, frame_idx)
            frame_idx += 1

            pct = int(frame_idx * 100 / max(1, frames_to_read))
            if pct >= last_log + 10:
                last_log = pct
                elapsed = time.time() - start_time
                fps_proc = frame_idx / max(1e-4, elapsed)
                print(f"Processing: {pct}% ({frame_idx}/{frames_to_read} frames) - {fps_proc:.1f} fps")
    finally:
        cap.release()

    events = pipeline.finalize(frame_idx - 1)
    elapsed_total = time.time() - start_time
    print(f"Inference completed in {elapsed_total:.2f}s ({frame_idx / max(1e-4, elapsed_total):.1f} fps)")
    print(f"Total counting events emitted: {len(events)}")

    # Export CSV
    csv_file = export_track1_csv(events, args.output_csv)
    print(f"Exported Track 1 counting results to: {csv_file}")

    # Evaluate if ground truth provided
    if args.ground_truth and args.ground_truth.is_file():
        print("-" * 60)
        print("Evaluating results against reference ground truth sample...")
        metrics = evaluate_track1_predictions(csv_file, args.ground_truth)
        metrics["video"] = {
            "name": args.video.name,
            "frames_processed": frame_idx,
            "duration_seconds": round(frame_idx / fps, 2),
            "fps": fps,
        }
        metrics["runtime_seconds"] = round(elapsed_total, 2)
        metrics["detector_model"] = str(args.detector)

        print(f"Ground Truth Total: {metrics['total_ground_truth']} (Car: {metrics['ground_truth_classes']['car']}, Truck: {metrics['ground_truth_classes']['truck']})")
        print(f"Predicted Total:    {metrics['total_predicted']} (Car: {metrics['predicted_classes']['car']}, Truck: {metrics['predicted_classes']['truck']})")
        print(f"Count Difference:   {metrics['overall_count_difference']}")
        print(f"MAE per Movement:   {metrics['mae_per_movement']}")
        print(f"RMSE per Movement:  {metrics['rmse_per_movement']}")
        print(f"Temporal Precision: {metrics['precision']:.4f}")
        print(f"Temporal Recall:    {metrics['recall']:.4f}")
        print(f"Temporal F1 Score:  {metrics['f1_score']:.4f}")

        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved evaluation report to: {args.metrics_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
