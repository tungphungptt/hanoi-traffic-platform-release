#!/usr/bin/env python3
"""Run AICC inference on a local video and emit a concise machine-readable report.

This tool verifies a real local clip passes through the AICC detector/tracker/classifier.
AICC v1 is a two-class video classifier; it is not the separate Hanoi +5/+10/+15 forecaster.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def inspect_video(path: Path, *, max_duration_seconds: int, max_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Video file not found: {path}")
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported video extension; use MP4/AVI/MOV/MKV/WEBM/M4V")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"Video size must be between 1 byte and {max_bytes // (1024*1024)} MB")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV cannot open this video/codec")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video metadata is incomplete (fps/frame_count/width/height)")
        duration = frame_count / fps
        if duration > max_duration_seconds:
            raise ValueError(f"Video duration {duration:.1f}s exceeds {max_duration_seconds}s cap")
        return {"size_bytes": size, "fps": fps, "frame_count": frame_count, "width": width,
                "height": height, "duration_seconds": duration}
    finally:
        cap.release()


def build_summary(video_path: Path, video_meta: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("results", [])
    counts = Counter(str(row.get("congestion_class", "unknown")) for row in rows if isinstance(row, dict))
    v = result.get("video", {})
    return {
        "report_version": "aicc-video-smoke/1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": video_path.name,
        "input_video_metadata": video_meta,
        "status": result.get("status"),
        "pipeline": result.get("pipeline"),
        "model_version": result.get("model_version"),
        "video_result_count": result.get("results_count"),
        "class_counts": dict(counts),
        "sample_fps": v.get("detector_sample_fps"),
        "frame_stride": v.get("frame_stride"),
        "frames_sampled": v.get("frames_sampled"),
        "video_result": result,
        "interpretation_limit": "AICC v1 is a 2-class sequence classifier and uses pixel/second speed; it is not a calibrated Hanoi forecast at +5/+10/+15 minutes.",
    }


def main() -> int:
    from aicc.pipeline import MAX_VIDEO_BYTES, process_video, aicc_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a permitted local video file")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Detector sampling FPS (clamped by pipeline)")
    parser.add_argument("--max-duration-seconds", type=int, default=1200, help="Maximum accepted video duration")
    parser.add_argument("--output", type=Path, help="Optional path to write a JSON diagnostic report")
    args = parser.parse_args()
    try:
        if args.max_duration_seconds < 1 or args.max_duration_seconds > 1200:
            raise ValueError("max-duration-seconds must be between 1 and 1200")
        video = args.video.expanduser().resolve()
        metadata = inspect_video(video, max_duration_seconds=args.max_duration_seconds, max_bytes=MAX_VIDEO_BYTES)
        bundle = aicc_runtime.load(force=True)
        last_bucket = -1

        def progress(current: int, total: int) -> None:
            nonlocal last_bucket
            if total <= 0:
                return
            bucket = min(10, int(current * 10 / total))
            if bucket > last_bucket:
                last_bucket = bucket
                print(f"Inference: {bucket * 10}%", file=sys.stderr, flush=True)

        result = process_video(video, bundle, progress_callback=progress, sample_fps_target=args.sample_fps)
        v = result.get("video", {})
        print(f"Sampling configuration: sample_fps={v.get('detector_sample_fps')}, frame_stride={v.get('frame_stride')}, frames_sampled={v.get('frames_sampled')}/{v.get('frames_processed')}", file=sys.stderr, flush=True)
        report = build_summary(video, metadata, result)
        output = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            target = args.output.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output + "\n", encoding="utf-8")
            print(f"Report written: {target}", file=sys.stderr)
        print(output)
        return 0
    except Exception as exc:
        print(f"VIDEO_INFERENCE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
