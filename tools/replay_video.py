#!/usr/bin/env python3
"""Simulated pilot: play recorded video as if a camera had been streaming, at full speed.

Each video is measured like a live camera (same ``FrameMeter`` as ``live_collector.py``) and the 1 Hz
samples are written to the history file with a clock that starts at ``--start``. Videos are laid
end to end, so consecutive parts of one recording behave as one long stream. No weather is attached:
a recording has no known date, so any weather would be made up.

    python tools/replay_video.py part000.mp4 part001.mp4 --camera-id demo-tdh --db live.sqlite \\
        --start "2026-03-02 07:00" --name "Trần Duy Hưng (mô phỏng)" --lat 21.0075 --lng 105.7993 --forecast

``--forecast`` finishes by asking the live store for the forecast at the end of the replay.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.history import LOCAL_UTC_OFFSET, History  # noqa: E402
from traffic_brain.live import LiveStore  # noqa: E402
from traffic_brain.minute_forecaster import record_to_series  # noqa: E402
from traffic_brain.video_analysis import FrameMeter  # noqa: E402


def parse_start(text: str) -> float:
    """``2026-03-02 07:00`` in Hanoi time, or a Unix epoch."""
    try:
        return float(text)
    except ValueError:
        local = datetime.strptime(text, "%Y-%m-%d %H:%M")
        return (local.replace(tzinfo=timezone.utc) - timedelta(seconds=LOCAL_UTC_OFFSET)).timestamp()


def replay(videos: list[Path], detector: Detector, history: History, camera_id: str, start_epoch: float, *,
           sample_fps: float = 1.0, name: str | None = None, lat: float | None = None, lng: float | None = None) -> dict:
    history.save_camera(camera_id, name or camera_id, lat, lng)
    offset, written, skipped = 0.0, 0, 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"Không mở được {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        stride = max(1, int(round(fps / max(0.25, min(2.0, sample_fps)))))
        meter = FrameMeter(detector, fps / stride)
        rows, index, last_media = [], 0, -1e9
        try:
            while True:
                if index % stride:
                    if not cap.grab():
                        break
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    media = index / fps
                    if media - last_media >= 0.5 / sample_fps:
                        last_media = media
                        row, _, banner = meter.measure(frame, media)
                        if banner:
                            skipped += 1
                        else:
                            rows.append({"t": round(start_epoch + offset + media, 3), "raw_ci": row["raw_ci"],
                                         "vehicle_count": row["vehicle_count"], "occupancy_ratio": row["occupancy_ratio"],
                                         "normalized_speed": row.get("normalized_speed"), "turnover": row.get("turnover")})
                index += 1
            duration = index / fps
        finally:
            cap.release()
        written += history.add_samples(camera_id, rows)
        offset += duration
        print(f"  {video.name}: {len(rows)} mẫu", flush=True)
    return {"written": written, "skipped_non_traffic": skipped, "end_epoch": start_epoch + offset}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--start", required=True, help='Hanoi time "YYYY-MM-DD HH:MM" or a Unix epoch')
    parser.add_argument("--name")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--forecast", action="store_true", help="print the live forecast at the end of the replay")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "detector.onnx")
    args = parser.parse_args()
    detector = Detector(args.model)
    detector.load()
    history = History(args.db)
    result = replay(args.videos, detector, history, args.camera_id, parse_start(args.start),
                    sample_fps=args.sample_fps, name=args.name, lat=args.lat, lng=args.lng)
    print(f"Đã ghi {result['written']} mẫu (bỏ {result['skipped_non_traffic']} khung không phải đường). {history.counts()}")
    if args.forecast:
        library = [s for s in (record_to_series(json.loads(f.read_text(encoding="utf-8")))
                               for f in sorted((ROOT / "models" / "forecast_library").glob("*.json"))) if s is not None]
        store = LiveStore(lambda: (library, len(library)), clock=lambda: result["end_epoch"] + 1, history=history)
        store.restore()
        found = store.forecast(args.camera_id) or {}
        print({k: found.get(k) for k in ("status", "samples", "observed_ci", "level_key", "horizons")})


if __name__ == "__main__":
    main()
