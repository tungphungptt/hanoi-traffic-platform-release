#!/usr/bin/env python3
"""Build the forecaster's training library from long traffic videos and check it.

Each video goes through the same pipeline as the web app (detector -> visual CI ->
per-camera series). One record per camera is written to --out. Then every record is
forecast using only records from *other* videos (leave-one-video-out) and scored against
what the video later showed.

    python tools/build_forecast_library.py a.mp4 b.mp4 --out models/forecast_library \\
        --source-note "where the videos come from" --demo held_out.mp4

--demo analyses one extra video with the whole library (it is never added to the
library) and writes a session file the web page can open next to the video.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from traffic_brain import minute_forecaster as mf  # noqa: E402
from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.video_analysis import analyze_video, result_payload  # noqa: E402

SESSION_CONTRACT = "hanoiflow-forecast-session/1.1"


def _analyse(path: Path, detector: Detector, library=(), library_videos: int = 0) -> dict:
    started = time.time()

    def progress(pct: int, _done: int, _total: int) -> None:
        if pct % 10 == 0:
            print(f"  {path.name}: {pct}%", end="\r", flush=True)
    analysis = analyze_video(path, detector, max_duration_seconds=4 * 3600, library=library,
                             library_videos=library_videos, progress=progress)
    print(f"  {path.name}: {analysis['mode']}, {len(analysis['cameras'])} camera(s), {time.time() - started:.0f}s")
    return analysis


def records_for(path: Path, analysis: dict, source_note: str) -> list[dict]:
    key = mf.video_key(path)
    out = []
    for cam in analysis["cameras"]:
        series = cam["series"]
        observed = int(np.isfinite(series["ci"]).sum())
        if observed < 20:
            continue
        record_key = key if len(analysis["cameras"]) == 1 else f"{key}-cam{cam['id']}"
        record = mf.series_record(series, video_key=record_key, filename=path.name)
        record.update(source_video_keys=[key], camera=cam["label"], mode=analysis["mode"],
                      observed_minutes=round(observed * mf.GRID_STEP_SECONDS / 60, 1),
                      duration_seconds=analysis["video"]["duration_seconds"], source_note=source_note)
        out.append(record)
    return out


def bootstrap_skill(per_source: dict, horizon: int, rounds: int = 2000, seed: int = 7) -> list[float] | None:
    """95% interval of the improvement over persistence, resampling whole videos."""
    groups = [v[horizon] for v in per_source.values() if v[horizon][0] > 0]
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(groups, dtype=np.float64)  # rows: (n, sum model err, sum persistence err)
    skills = []
    for _ in range(rounds):
        pick = data[rng.integers(0, len(data), len(data))]
        if pick[:, 2].sum() > 0:
            skills.append(1 - pick[:, 1].sum() / pick[:, 2].sum())
    lo, hi = np.percentile(skills, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def leave_one_video_out(records: list[dict]) -> dict:
    totals = {h: {"n": 0, "model": 0.0, "persistence": 0.0, "trend": 0.0} for h in mf.HORIZONS_MINUTES}
    per_record = []
    per_source: dict = {}
    series = [mf.record_to_series(r) for r in records]
    for i, record in enumerate(records):
        own = set(record["source_video_keys"])
        library = [s for j, s in enumerate(series) if j != i and not own & set(records[j]["source_video_keys"])]
        metrics = mf.evaluate_anchors(mf.forecast_series(series[i], library=library))
        per_record.append({"video_key": record["video_key"], "filename": record["filename"],
                           "camera": record.get("camera"), "metrics": metrics})
        source = per_source.setdefault(tuple(sorted(own)), {h: [0, 0.0, 0.0] for h in mf.HORIZONS_MINUTES})
        for m in metrics:
            if m.get("evaluated"):
                g = source[m["horizon_minutes"]]
                g[0] += m["evaluated"]
                g[1] += m["mae_model"] * m["evaluated"]
                g[2] += m["mae_persistence"] * m["evaluated"]
                t = totals[m["horizon_minutes"]]
                t["n"] += m["evaluated"]
                for name in ("model", "persistence", "trend"):
                    t[name] += m[f"mae_{name}"] * m["evaluated"]
    summary = []
    for h, t in totals.items():
        if not t["n"]:
            summary.append({"horizon_minutes": h, "evaluated": 0})
            continue
        model, persistence, trend = (t[k] / t["n"] for k in ("model", "persistence", "trend"))
        summary.append({"horizon_minutes": h, "evaluated": t["n"], "mae_model": round(model, 4),
                        "mae_persistence": round(persistence, 4), "mae_trend": round(trend, 4),
                        "skill_vs_persistence": round(1 - model / persistence, 4) if persistence else None,
                        "skill_ci95": bootstrap_skill(per_source, h), "videos": sum(1 for v in per_source.values() if v[h][0])})
    return {"summary": summary, "records": per_record}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "forecast_library")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "forecast_library_report.json")
    parser.add_argument("--source-note", default="")
    parser.add_argument("--demo", type=Path, default=None)
    parser.add_argument("--demo-out", type=Path, default=None)
    args = parser.parse_args()

    detector = Detector()
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        analysis = _analyse(video, detector)
        for record in records_for(video, analysis, args.source_note):
            (args.out / f"{record['video_key']}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.out.glob("*.json"))]
    for r in records:
        r.setdefault("source_video_keys", [r["video_key"]])
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "records": len(records), **leave_one_video_out(records)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in report["summary"]:
        print(row)

    if args.demo:
        key = mf.video_key(args.demo)
        library = [mf.record_to_series(r) for r in records if key not in r["source_video_keys"]]
        sources = {tuple(r["source_video_keys"]) for r in records if key not in r["source_video_keys"]}
        analysis = _analyse(args.demo, detector, library, len(sources))
        session = {"contract": SESSION_CONTRACT, "created_at": report["created_at"],
                   **result_payload(analysis, args.demo.name)}
        target = args.demo_out or ROOT / "artifacts" / f"demo_{args.demo.stem}.hanoiflow.json"
        target.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
        for cam in session["cameras"]:
            print(cam["label"], cam["on_air_seconds"], [(m["horizon_minutes"], m.get("evaluated"), m.get("skill_vs_persistence"))
                                                        for m in cam["minute_forecast"].get("metrics", [])])
        print("demo session:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
