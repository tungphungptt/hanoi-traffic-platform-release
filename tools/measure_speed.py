#!/usr/bin/env python3
"""Measure vehicle speed (km/h) in a video, on a road plane calibrated for that camera.

Calibration file (JSON), four points on the road whose real distances you know, e.g. the corners of a
rectangle between two lane lines and two cross-lines, taken from the pixels of one frame and from a
tape or the ruler of a satellite map. World coordinates are in metres, X across the road, Y along it:

    {"image_points": [[412, 530], [655, 528], [590, 300], [470, 301]],
     "world_points_m": [[0, 0], [7, 0], [7, 40], [0, 40]]}

    python tools/measure_speed.py clip.mp4 --calib cam.json --out speeds.json --preview check.jpg

``--preview`` saves a frame with a 5 m grid: if the grid does not look like the road, fix the points
before trusting any number. Use --model yolox_s.onnx (see models/README.md) for a detector that sees
more vehicles. Speeds come from the vehicles inside the calibrated quadrilateral only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.speed import MAX_PLAUSIBLE_KMH, GroundPlane, SpeedTracker, draw_calibration, summarise  # noqa: E402


def measure(video: Path, detector, plane: GroundPlane, *, fps: float = 4.0, start: float = 0.0,
            max_seconds: float | None = None, window: float = 10.0) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được {video}")
    native = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    stride = max(1, int(round(native / fps)))
    first = int(start * native)
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker = SpeedTracker(plane)
    index, last_t = first, start
    try:
        while True:
            t = index / native
            if max_seconds is not None and t - start > max_seconds:
                break
            if (index - first) % stride:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                tracker.update(t, detector.detect(frame))
                last_t = t
            index += 1
    finally:
        cap.release()
    tracks = tracker.finish()
    rows = summarise(tracks, window_seconds=window, duration=last_t)
    speeds = np.array([tr.speed_kmh for tr in tracks])
    warning = None
    if len(speeds) and float(np.median(speeds)) > MAX_PLAUSIBLE_KMH:
        warning = "Tốc độ trung vị vượt 120 km/h: gần như chắc chắn calibration sai."
    return {"video": video.name, "sample_fps": native / stride, "seconds": round(last_t - start, 1), "tracks": len(tracks),
            "overall": {"median_kmh": round(float(np.median(speeds)), 1) if len(speeds) else None,
                        "mean_kmh": round(float(speeds.mean()), 1) if len(speeds) else None,
                        "p85_kmh": round(float(np.percentile(speeds, 85)), 1) if len(speeds) else None},
            "windows": rows, "warning": warning,
            "note": "Tốc độ từ homography 4 điểm; chỉ đúng khi calibration đúng và xe nằm trong tứ giác."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "detector.onnx")
    parser.add_argument("--fps", type=float, default=4.0, help="frames per second processed (3-6 is a good range)")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preview", type=Path, help="save a frame with the calibration grid and exit unless --out is given")
    args = parser.parse_args()
    plane = GroundPlane.from_file(args.calib)
    if args.preview:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Không đọc được khung hình để vẽ.")
        cv2.imwrite(str(args.preview), draw_calibration(frame, plane))
        print(f"Đã lưu {args.preview}")
        if not args.out:
            return
    detector = Detector(args.model)
    detector.load()
    result = measure(args.video, detector, plane, fps=args.fps, start=args.start, max_seconds=args.max_seconds, window=args.window)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("video", "seconds", "tracks", "overall", "warning")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
