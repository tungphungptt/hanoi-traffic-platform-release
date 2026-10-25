#!/usr/bin/env python3
"""CLI utility to generate YOLO pseudo-labels from video using the ONNX detector."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onnxruntime as ort

from aicc.auto_labeling import AutoLabelPipeline
from aicc.track1_counting import load_roi_polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4",
        help="Path to video file.",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt",
        help="Optional path to ROI polygon file.",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=ROOT / "models/detector.onnx",
        help="Path to detector ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/auto_labeling/cam_5_sample",
        help="Directory to save pseudo-labels and manifest.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=10,
        help="Frame sampling stride (e.g., 10 = 1 frame per second at 10 FPS).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames to process.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Whether to save extracted frame images alongside labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.is_file():
        print(f"[ERROR] Video file not found: {args.video}", file=sys.stderr)
        return 1
    if not args.detector.is_file():
        print(f"[ERROR] Detector not found: {args.detector}", file=sys.stderr)
        return 1

    roi_poly = None
    if args.roi and args.roi.is_file():
        roi_poly = load_roi_polygon(args.roi)

    session = ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])
    pipeline = AutoLabelPipeline(
        detector_session=session,
        roi_polygon=roi_poly,
        conf_threshold=args.conf_threshold,
    )

    print("=" * 60)
    print("AI City Track 1 - Pseudo-Label Generation Pipeline")
    print("=" * 60)
    print(f"Video:             {args.video}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Sampling stride:   {args.sample_stride}")
    print(f"Max frames:        {args.max_frames}")
    print(f"Confidence cutoff: {args.conf_threshold}")

    t0 = time.time()
    manifest = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
        save_images=args.save_images,
    )
    elapsed = time.time() - t0

    stats = manifest["summary_statistics"]
    print("-" * 60)
    print(f"Completed in {elapsed:.2f}s")
    print(f"Sampled frames:    {manifest['dataset_provenance']['sampled_frames_count']}")
    print(f"Total boxes:       {stats['total_pseudo_boxes']}")
    print(f"  - Cars:          {stats['total_cars']}")
    print(f"  - Trucks:        {stats['total_trucks']}")
    print(f"  - Inside ROI:    {stats['total_in_roi']}")
    print(f"Average boxes/fr:  {stats['average_boxes_per_frame']}")
    print(f"Manifest written:  {args.output_dir / 'auto_labels_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
