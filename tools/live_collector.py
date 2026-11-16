#!/usr/bin/env python3
"""Measure an authorised camera stream and send 1 Hz CI samples to a hanoiFlow server.

The server must have LIVE_TOKEN set; pass the same value with --token (or the LIVE_TOKEN
environment variable). Nothing is stored on disk. Use only streams you are allowed to read.

    python tools/live_collector.py --source rtsp://... --camera-id tdh-216 --name "216 Trần Duy Hưng" \
        --server https://your-app.onrender.com --lat 21.0075 --lng 105.7993

``--source`` may also be a camera index or a video file (played at its own speed), which is
how to try the live page without a camera.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.video_analysis import FrameMeter  # noqa: E402

MAX_BUFFER = 600  # samples kept while the server is unreachable (the server accepts up to 600 per post)


def sample_from_row(row: dict[str, Any], wall_time: float) -> dict[str, Any]:
    sample: dict[str, Any] = {"t": round(wall_time, 3), "ci": row["raw_ci"], "vehicle_count": row["vehicle_count"],
                              "occupancy_ratio": row["occupancy_ratio"]}
    if row.get("normalized_speed") is not None:
        sample["normalized_speed"] = row["normalized_speed"]
    if row.get("turnover") is not None:
        sample["turnover"] = row["turnover"]
    return sample


def measure_stream(source: str, detector: Detector, *, sample_fps: float = 1.0,
                   clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                   reconnect_seconds: float = 5.0, max_failures: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield one sample per ``1/sample_fps`` seconds. Streams are sampled on the wall clock;
    a file is sampled on its own timeline, paced to play at its own speed."""
    is_file = Path(source).is_file()
    failures = 0
    while True:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            cap.release()
            failures += 1
            if is_file or (max_failures is not None and failures >= max_failures):
                return
            sleep(reconnect_seconds)
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        stride = max(1, int(round(fps / max(0.25, min(2.0, sample_fps)))))
        meter = FrameMeter(detector, fps / stride)
        started, index, last_wall = clock(), 0, -1e9
        try:
            while True:
                if index % stride:
                    if not cap.grab():
                        break
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    media_time = index / fps
                    wall = started + media_time if is_file else clock()
                    if is_file:
                        wait = wall - clock()
                        if wait > 0:
                            sleep(wait)
                    if wall - last_wall >= 0.5 / sample_fps:
                        last_wall = wall
                        row, _, banner = meter.measure(frame, media_time)
                        if not banner:
                            yield sample_from_row(row, wall)
                index += 1
        finally:
            cap.release()
        if is_file:
            return
        failures += 1
        if max_failures is not None and failures >= max_failures:
            return
        sleep(reconnect_seconds)


def http_post(url: str, token: str, payload: dict[str, Any], timeout: float = 15.0) -> int:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-Live-Token": token})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def run(samples: Iterator[dict[str, Any]], post: Callable[[dict[str, Any]], Any], *,
        meta: dict[str, Any], batch_seconds: float = 5.0, clock: Callable[[], float] = time.time) -> int:
    """Send samples in batches; keep them (bounded) while ``post`` fails. Returns samples sent."""
    pending: deque[dict[str, Any]] = deque(maxlen=MAX_BUFFER)
    sent, last_post = 0, clock()

    def flush() -> None:
        nonlocal sent, last_post
        last_post = clock()
        if not pending:
            return
        batch = list(pending)
        try:
            post({**meta, "samples": batch})
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 503):
                raise SystemExit(f"Máy chủ từ chối ({exc.code}): kiểm tra LIVE_TOKEN và cấu hình máy chủ.") from exc
            if exc.code == 422:
                pending.clear()  # a batch the server will never accept
                print("Máy chủ từ chối gói số liệu:", exc.read().decode("utf-8", "replace")[:200], file=sys.stderr)
            return
        except (urllib.error.URLError, OSError, TimeoutError):
            return  # keep the batch, retry on the next flush
        pending.clear()
        sent += len(batch)

    for sample in samples:
        pending.append(sample)
        if clock() - last_post >= batch_seconds:
            flush()
    flush()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="RTSP/HLS/HTTP URL, camera index or a video file")
    parser.add_argument("--camera-id", required=True, help="lowercase letters, digits, - or _")
    parser.add_argument("--server", required=True, help="base URL of the hanoiFlow server")
    parser.add_argument("--token", default=os.getenv("LIVE_TOKEN", ""), help="LIVE_TOKEN of the server")
    parser.add_argument("--name")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--batch-seconds", type=float, default=5.0)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "detector.onnx")
    args = parser.parse_args()
    if not args.token:
        parser.error("cần --token hoặc biến môi trường LIVE_TOKEN")
    detector = Detector(args.model)
    detector.load()
    if not detector.available:
        sys.exit(f"Không nạp được mô hình nhận diện: {detector.error}")
    url = f"{args.server.rstrip('/')}/api/live/{args.camera_id}/samples"
    meta = {k: v for k, v in (("name", args.name), ("lat", args.lat), ("lng", args.lng)) if v is not None}
    total = run(measure_stream(args.source, detector, sample_fps=args.sample_fps),
                lambda payload: http_post(url, args.token, payload), meta=meta, batch_seconds=args.batch_seconds)
    print(f"Đã gửi {total} mẫu.")


if __name__ == "__main__":
    main()
