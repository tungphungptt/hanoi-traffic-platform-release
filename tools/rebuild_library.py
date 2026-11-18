#!/usr/bin/env python3
"""Rebuild the forecast library with one detector, then write every demo session from the same pass.

Each video is run through the detector once (cached as a pickle). When all are done, every demo is
forecast with the library minus its own video (matched by content key), so no video forecasts itself.

    python tools/rebuild_library.py detect NAME PATH [--tiles 2x1] [--model yolox_s.onnx]
    python tools/rebuild_library.py finish --cache DIR --manifest manifest.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_forecast_library import SESSION_CONTRACT, _analyse, leave_one_video_out, records_for  # noqa: E402
from traffic_brain import minute_forecaster as mf  # noqa: E402
from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.video_analysis import _camera_series, build_minute_forecast, result_payload, visual_ci  # noqa: E402


def detect(args: argparse.Namespace) -> None:
    cols, rows = (int(v) for v in args.tiles.split("x"))
    detector = Detector(Path(args.model), (cols, rows))
    analysis = _analyse(args.path, detector)
    analysis["_key"] = mf.video_key(args.path)
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / f"{args.name}.pkl").write_bytes(pickle.dumps({"analysis": analysis, "path": str(args.path)}))


def finish(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))  # [{name, in_library, session_out}]
    loaded = {m["name"]: pickle.loads((args.cache / f"{m['name']}.pkl").read_bytes()) for m in manifest}
    for m in manifest:  # optional camera names, in on-air order of the analysis
        for cam, label in zip(loaded[m["name"]]["analysis"]["cameras"], m.get("labels", [])):
            cam["label"] = label
    for item in loaded.values():  # re-score the cached detections with the current CI scale
        analysis = item["analysis"]
        for row in analysis["timeline"]:
            row["raw_ci"] = visual_ci(row["vehicle_count"], row["occupancy_ratio"], row["normalized_speed"])
        for cam in analysis["cameras"]:
            cam["series"] = _camera_series(analysis["timeline"], None if analysis["mode"] == "single_camera" else cam["id"])
    out = args.out
    for old in out.glob("*.json"):
        old.unlink()
    records = []
    for m in manifest:
        if not m.get("in_library", True):
            continue
        item = loaded[m["name"]]
        for record in records_for(Path(item["path"]), item["analysis"], args.source_note):
            (out / f"{record['video_key']}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            records.append(record)
    for r in records:
        r.setdefault("source_video_keys", [r["video_key"]])
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "records": len(records), **leave_one_video_out(records)}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in report["summary"]:
        print(row)
    for m in manifest:
        analysis = loaded[m["name"]]["analysis"]
        key = analysis["_key"]
        kept = [r for r in records if key not in r["source_video_keys"]]
        library = [mf.record_to_series(r) for r in kept]
        sources = len({tuple(r["source_video_keys"]) for r in kept})
        duration = analysis["video"]["duration_seconds"]
        for cam in analysis["cameras"]:
            cam["minute_forecast"] = build_minute_forecast((), duration_seconds=duration, timing_confirmed=False,
                                                           library=library, library_videos=sources, series=cam["series"])
            anchors = cam["minute_forecast"].get("anchors") or []
            cam["checked_forecasts"] = sum(1 for a in anchors for h in a["horizons"] if h.get("actual_ci") is not None)
        session = {"contract": SESSION_CONTRACT, "created_at": report["created_at"],
                   **result_payload(analysis, Path(loaded[m["name"]]["path"]).name)}
        Path(m["session_out"]).write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
        cam = session["cameras"][0]
        print(m["name"], [(x["horizon_minutes"], x.get("skill_vs_persistence")) for x in cam["minute_forecast"].get("metrics", [])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("name"); d.add_argument("path", type=Path)
    d.add_argument("--tiles", default="2x1"); d.add_argument("--model", default=str(ROOT / "models" / "yolox_s.onnx"))
    d.add_argument("--cache", type=Path, default=Path("/tmp/rebuild_cache"))
    f = sub.add_parser("finish")
    f.add_argument("--cache", type=Path, default=Path("/tmp/rebuild_cache")); f.add_argument("--manifest", type=Path, required=True)
    f.add_argument("--out", type=Path, default=ROOT / "models" / "forecast_library")
    f.add_argument("--report", type=Path, default=ROOT / "artifacts" / "forecast_library_report.json")
    f.add_argument("--source-note", default="")
    args = parser.parse_args()
    detect(args) if args.cmd == "detect" else finish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
