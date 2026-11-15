#!/usr/bin/env python3
"""Train the +5/+10/+15 minute forecaster from long traffic videos.

The minute forecaster learns from (situation now, CI change h minutes later) pairs.
This tool turns each video into an observed CI series with the same detector and
visual-CI rule the web app uses, stores one library record per video, then checks
the result with leave-one-video-out: every video is forecast using only the
*other* videos as training data, and scored against what actually happened.

Usage:
    python tools/build_forecast_library.py VIDEO [VIDEO ...] \
        --out models/forecast_library --report artifacts/evaluation/forecast_library_report.json

ONNX Runtime is used when installed; otherwise the detector runs through OpenCV DNN.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aicc.pipeline import COCO_CLASS_IDS  # noqa: E402
from traffic_brain import minute_forecaster as mf  # noqa: E402
from traffic_brain import video_review  # noqa: E402

DETECTOR = ROOT / "models" / "detector.onnx"


class _CvDnnSession:
    """Minimal ``onnxruntime.InferenceSession`` stand-in backed by OpenCV DNN."""

    def __init__(self, path: Path) -> None:
        import cv2
        self._net = cv2.dnn.readNetFromONNX(str(path))

    def run(self, _outputs, feeds):
        (tensor,) = feeds.values()
        self._net.setInput(np.ascontiguousarray(tensor, dtype=np.float32))
        return [self._net.forward()]


def load_detector() -> tuple[object, str, str]:
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(DETECTOR), providers=["CPUExecutionProvider"])
        return session, session.get_inputs()[0].name, "onnxruntime"
    except ImportError:
        return _CvDnnSession(DETECTOR), "images", "opencv-dnn"


def _no_second_level_lstm(_sequence):
    return {"available": False, "mode": "skipped_for_library_build", "predictions": []}


def analyse(path: Path, detector: object, input_name: str, sample_fps: float) -> dict:
    """Run the web app's video analysis and return its timeline and duration."""
    bundle = {
        "manifest": {"sequence_length": 10**9},  # skip the 2-class AICC head; not needed here
        "class_ids": set(COCO_CLASS_IDS.values()),
        "detector": detector,
        "detector_input_name": input_name,
    }
    started = time.time()
    last = [-1]

    def progress(percent: int, _done: int, _total: int) -> None:
        if percent // 10 != last[0]:
            last[0] = percent // 10
            print(f"  {path.name}: {percent}%", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        result = video_review.analyze_video(path, bundle, Path(tmp), preserve_real_timing=True,
                                            sample_fps=sample_fps, progress_callback=progress)
    return {"timeline": result["timeline"], "duration": result["video"]["duration_seconds"],
            "seconds": round(time.time() - started, 1)}


def leave_one_out(records: list[dict]) -> list[dict]:
    """Forecast each video with only the other videos as training library."""
    report = []
    for index, record in enumerate(records):
        series = mf.record_to_series(record)
        library = [mf.record_to_series(other) for j, other in enumerate(records) if j != index]
        library = [item for item in library if item is not None]
        anchors = mf.forecast_series(series, library=library)
        report.append({
            "video": record["filename"],
            "library_videos": len(library),
            "anchors": len(anchors),
            "metrics": mf.evaluate_anchors(anchors),
            "highlights": {key: len(value) for key, value in mf.select_highlights(anchors).items()},
        })
    return report


def summarise(report: list[dict]) -> list[dict]:
    rows = []
    for h in mf.HORIZONS_MINUTES:
        model, persistence, trend, n = [], [], [], 0
        for item in report:
            metric = next(m for m in item["metrics"] if m["horizon_minutes"] == h)
            if metric.get("evaluated"):
                weight = metric["evaluated"]
                model.append(metric["mae_model"] * weight)
                persistence.append(metric["mae_persistence"] * weight)
                trend.append(metric["mae_trend"] * weight)
                n += weight
        if n:
            mae_model, mae_persist, mae_trend = sum(model) / n, sum(persistence) / n, sum(trend) / n
            rows.append({"horizon_minutes": h, "evaluated": n, "mae_model": mae_model,
                         "mae_persistence": mae_persist, "mae_trend": mae_trend,
                         "skill_vs_persistence": 1 - mae_model / mae_persist if mae_persist > 1e-9 else None})
        else:
            rows.append({"horizon_minutes": h, "evaluated": 0})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "forecast_library")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "evaluation" / "forecast_library_report.json")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--source-note", default="", help="Provenance/licence note stored in each record")
    parser.add_argument("--demo", default=None, help="Filename of one video to export as a ready-to-open web session")
    parser.add_argument("--demo-out", type=Path, default=None, help="Where to write the demo session JSON")
    args = parser.parse_args()

    detector, input_name, backend = load_detector()
    # The second-level Hanoi LSTM is unrelated to minute forecasts; skip it here.
    video_review.hanoi_video_forecaster.predict = _no_second_level_lstm
    print(f"Detector backend: {backend}")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.videos:
        key = mf.video_key(path)
        target = args.out / f"{key}.json"
        if target.is_file():
            print(f"{path.name}: dùng lại {target.name}")
            records.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        print(f"{path.name}: đang phân tích…")
        analysed = analyse(path, detector, input_name, args.sample_fps)
        series = mf.build_observed_series(analysed["timeline"])
        record = mf.series_record(series, video_key=key, filename=path.name)
        record.update({
            "duration_seconds": analysed["duration"],
            "detector_backend": backend,
            "source_note": args.source_note,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        target.write_text(json.dumps(record), encoding="utf-8")
        ci = np.asarray([c for c in record["ci"] if c is not None])
        print(f"  xong sau {analysed['seconds']} s · CI TB {ci.mean():.3f} · min {ci.min():.3f} · max {ci.max():.3f}")
        records.append(record)

    report = leave_one_out(records) if len(records) > 1 else []
    summary = summarise(report) if report else []
    payload = {
        "contract": "hanoiflow-forecast-library-report/1.0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "leave-one-video-out: each video forecast with only the other videos as library; "
                    "in-video learning is still causal (only data before each anchor).",
        "warning_probability": mf.WARNING_PROBABILITY,
        "videos": [{"filename": r["filename"], "video_key": r["video_key"],
                    "duration_seconds": r.get("duration_seconds"), "source_note": r.get("source_note", "")}
                   for r in records],
        "per_video": report,
        "summary": summary,
    }
    if args.demo:
        demo = next((r for r in records if r["filename"] == args.demo), None)
        if demo is None:
            raise SystemExit(f"--demo {args.demo!r} is not one of the analysed videos")
        library = [mf.record_to_series(r) for r in records if r is not demo]
        library = [item for item in library if item is not None]
        forecast = mf.build_minute_forecast([], duration_seconds=float(demo.get("duration_seconds") or 0),
                                            timing_confirmed=True, library=library,
                                            library_videos=len(library), series=mf.record_to_series(demo))
        session = {
            "contract": "hanoiflow-forecast-session/1.0",
            "created_at": payload["created_at"],
            "filename": demo["filename"],
            "video": {"duration_seconds": demo.get("duration_seconds")},
            "minute_forecast": forecast,
            "note": "Library = the other videos only (leave-one-out). Open on the home page with "
                    "'Mở kết quả đã lưu', then choose the same video file.",
        }
        out = args.demo_out or (args.report.parent / f"demo_{Path(demo['filename']).stem}.hanoiflow.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
        print(f"Demo session: {out}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summary:
        if row.get("evaluated"):
            print(f"+{row['horizon_minutes']} phút: n={row['evaluated']} MAE mô hình {row['mae_model']:.3f} · "
                  f"giữ nguyên {row['mae_persistence']:.3f} · xu hướng {row['mae_trend']:.3f} · "
                  f"cải thiện {row['skill_vs_persistence'] * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
