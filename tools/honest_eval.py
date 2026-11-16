#!/usr/bin/env python3
"""Honest check of the jam warnings: pick every threshold WITHOUT looking at the video being scored.

The forecast library check (``build_forecast_library.py``) forecasts each record from the other
videos, but the warning thresholds (40% warning, 20% watch) were set after seeing its results, so
its onset counts are optimistic. Here the threshold is chosen again for every held-out video, using
only the other videos' forecasts (nested leave-one-video-out), and the held-out video is scored
with that threshold.

    python tools/honest_eval.py --out artifacts/honest_eval.json

An onset is a forecast made while CI < gridlock whose target time is at or above gridlock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from traffic_brain import minute_forecaster as mf  # noqa: E402

HORIZONS = (5, 10, 15)
GRID = [round(float(g), 2) for g in np.arange(0.02, 0.62, 0.02)]
MIN_PRECISION = 0.5  # a warning level should be right at least half of the time it fires


def loo_rows(library_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(library_dir.glob("*.json"))]
    for r in records:
        r.setdefault("source_video_keys", [r["video_key"]])
    series = [mf.record_to_series(r) for r in records]
    rows = []
    for i, rec in enumerate(records):
        own = set(rec["source_video_keys"])
        library = [s for j, s in enumerate(series) if j != i and not own & set(records[j]["source_video_keys"])]
        for anchor in mf.forecast_series(series[i], library=library):
            for item in anchor["horizons"]:
                if item["actual_ci"] is None:
                    continue
                rows.append({"src": rec["source_video_keys"][0], "h": item["horizon_minutes"], "now": float(anchor["observed_ci"]),
                             "act": float(item["actual_ci"]), "p": float(item["p_gridlock"]), "pred": float(item["ci"])})
    return rows


def _counts(rows: list[dict], threshold: float) -> tuple[int, int, int]:
    tp = sum(r["p"] >= threshold and r["act"] >= mf.SEVERE_CI for r in rows)
    fp = sum(r["p"] >= threshold and r["act"] < mf.SEVERE_CI for r in rows)
    pos = sum(r["act"] >= mf.SEVERE_CI for r in rows)
    return int(tp), int(fp), int(pos)


def _pick(train: list[dict], mode: str) -> float | None:
    """warning: the lowest threshold whose precision on the other videos is >= MIN_PRECISION, and that catches something.
    watch: the threshold with the best F1 on the other videos."""
    best, best_score = None, 0.0
    for g in GRID:
        tp, fp, pos = _counts(train, g)
        if tp == 0 or pos == 0:
            continue
        precision, recall = tp / (tp + fp), tp / pos
        if mode == "warning":
            if precision >= MIN_PRECISION and recall > best_score:
                best, best_score = g, recall
        else:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_score:
                best, best_score = g, f1
    return best


def nested(rows: list[dict]) -> dict:
    out = {}
    for h in HORIZONS:
        pool = [r for r in rows if r["h"] == h and r["now"] < mf.SEVERE_CI]
        sources = sorted({r["src"] for r in pool})
        item = {"forecasts": len(pool), "onsets": int(sum(r["act"] >= mf.SEVERE_CI for r in pool)),
                "onset_videos": len({r["src"] for r in pool if r["act"] >= mf.SEVERE_CI})}
        for mode in ("warning", "watch"):
            tp = fp = pos = 0
            chosen = []
            for src in sources:
                train = [r for r in pool if r["src"] != src]
                test = [r for r in pool if r["src"] == src]
                threshold = _pick(train, mode)
                chosen.append(threshold)
                if threshold is None:
                    pos += sum(r["act"] >= mf.SEVERE_CI for r in test)
                    continue
                a, b, c = _counts(test, threshold)
                tp, fp, pos = tp + a, fp + b, pos + c
            item[mode] = {"caught": tp, "false": fp, "onsets": pos,
                          "recall": round(tp / pos, 3) if pos else None,
                          "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                          "thresholds_chosen": sorted({c for c in chosen if c is not None}),
                          "folds_without_threshold": sum(c is None for c in chosen)}
        out[f"+{h}"] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=ROOT / "models" / "forecast_library")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "honest_eval.json")
    args = parser.parse_args()
    rows = loo_rows(args.library)
    result = nested(rows)
    payload = {"note": "thresholds chosen on the other videos only; MIN_PRECISION=%.2f for warning, best F1 for watch" % MIN_PRECISION,
               "horizons": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, item in result.items():
        w, t = item["warning"], item["watch"]
        print(f"{key}: onsets {item['onsets']} in {item['onset_videos']} videos | warning {w['caught']}/{w['onsets']} "
              f"false {w['false']} thr {w['thresholds_chosen']} | watch {t['caught']}/{t['onsets']} false {t['false']} thr {t['thresholds_chosen']}")


if __name__ == "__main__":
    main()
