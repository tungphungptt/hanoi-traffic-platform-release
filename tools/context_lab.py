#!/usr/bin/env python3
"""Do the time of week and the rain improve the forecast? Answerable only with real, dated data.

Input: the CSV from ``GET /api/live/export`` (CI with Hanoi clock time and the weather snapshot at
the time). For every camera, leave-one-day-out: the "typical CI of this weekday and hour" and the
rain flags are learned from the other days only, added as extra formulas to the shipped three, and
the extra weights are fitted by non-negative least squares. The extras are adopted only if the MAE
gain over the shipped formulas has a 95% interval above zero (bootstrap over whole days) and at least
``--min-days`` days exist, so a pattern seen on a handful of days is not mistaken for a rule.

    python tools/context_lab.py live_export.csv --out artifacts/context_lab.json

The comparison base is the shipped three formulas plus "level_gap" (the camera's usual CI from the other days
minus CI now): knowing a camera's level from earlier days helps by itself, and must not be credited to the clock.
Extra formulas for horizon h (all >= 0 weights, so they can only act in the stated direction):
  typical_gap     typical CI at the target time minus CI now (go toward what this hour usually looks like)
  typical_change  typical CI at the target time minus typical CI now (the usual rise into rush hour)
  rain_onset      1 when it rains now but did not 5 minutes ago (traffic slows in the first minutes of rain;
                  a rain that has lasted for hours is already part of CI and needs no formula)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from traffic_brain import minute_forecaster as mf  # noqa: E402

HORIZONS = (5, 10, 15)
STEPS = int(round(60.0 / mf.GRID_STEP_SECONDS))
LOCAL = 7 * 3600
RAIN_MM = 0.1
EXTRAS = ("typical_gap", "typical_change", "rain_onset")
RAIN_ONSET_MINUTES = 5
BOOTSTRAPS = 2000
# The shipped ridge (1.0) shrinks weights less when a formula is duplicated, which would credit extra
# columns with a gain that has nothing to do with them; both models are fitted without it here.
NO_RIDGE = 1e-9


def read_export(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per camera: grid series plus local weekday/hour and rain flag at every grid point."""
    rows: dict[str, list] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rain = r.get("precipitation_mm")
            rows[r["camera_id"]].append((float(r["t_epoch"]), float(r["ci"]), float(rain) if rain not in (None, "") else math.nan))
    out = {}
    for camera, items in rows.items():
        items.sort()
        origin = items[0][0]
        series = mf.build_observed_series([{"timestamp_seconds": t - origin, "raw_ci": ci} for t, ci, _ in items])
        epoch = origin + series["t"]
        local = epoch + LOCAL
        rain_t = np.array([x[0] for x in items])
        rain_v = np.array([x[2] for x in items])
        pos = np.clip(np.searchsorted(rain_t, epoch, side="right") - 1, 0, len(rain_v) - 1)
        rain = np.where(np.isfinite(rain_v[pos]), rain_v[pos] >= RAIN_MM, np.nan)
        out[camera] = {"series": series, "epoch": epoch, "slot": (((local // 86400 + 3) % 7) * 24 + ((local % 86400) // 3600)).astype(int),
                       "day": (local // 86400).astype(int), "rain": rain,
                     "feats": mf.compute_features(series)}
    return out


def typical_table(ci: np.ndarray, slot: np.ndarray, mask: np.ndarray) -> np.ndarray:
    table = np.full(168, np.nan)
    for s in range(168):
        sel = mask & (slot == s) & np.isfinite(ci)
        if sel.sum() >= 20:
            table[s] = ci[sel].mean()
    return table


def design(cam: dict, table: np.ndarray, h: int, idx: np.ndarray, with_extras: bool) -> np.ndarray:
    series = cam["series"]
    feats = cam["feats"]
    base = mf._component_matrix(feats[idx], h)
    k = h * STEPS
    ci = np.asarray(series["ci"], dtype=float)
    slot = cam["slot"]
    target_slot = slot[np.minimum(idx + k, len(slot) - 1)]
    typ_target, typ_now = table[target_slot], table[slot[idx]]
    level_gap = np.nan_to_num(table_mean(table) - ci[idx])  # the camera's own usual level, known from other days
    rain = cam["rain"]
    now = np.nan_to_num(rain[idx].astype(float), nan=0.0)
    before = np.nan_to_num(rain[np.maximum(idx - RAIN_ONSET_MINUTES * STEPS, 0)].astype(float), nan=0.0)
    extras = np.column_stack([typ_target - ci[idx], typ_target - typ_now, now * (1.0 - before)])
    return np.hstack([base, level_gap[:, None], np.nan_to_num(extras)])


def table_mean(table: np.ndarray) -> float:
    return float(np.nanmean(table)) if np.isfinite(table).any() else 0.0


def valid_index(cam: dict, h: int) -> np.ndarray:
    series, k = cam["series"], h * STEPS
    ci = np.asarray(series["ci"], dtype=float)
    feats = cam["feats"]
    idx = np.flatnonzero(np.all(np.isfinite(feats), axis=1) & (np.asarray(series["t"]) >= mf.WARMUP_SECONDS))
    idx = idx[idx + k < len(ci)]
    return idx[np.isfinite(ci[idx + k])]


def evaluate(data: dict[str, dict], min_days: int) -> dict:
    days = sorted({int(d) for cam in data.values() for d in np.unique(cam["day"])})
    result: dict = {"days": len(days), "cameras": len(data), "min_days": min_days}
    if len(days) < min_days:
        result["verdict"] = f"Chưa đủ dữ liệu: có {len(days)} ngày, cần ít nhất {min_days}."
        return result
    rng = np.random.default_rng(5)
    for h in HORIZONS:
        per_day: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
        for held in days:
            train_parts, test_parts = [], []
            for cam in data.values():
                ci = np.asarray(cam["series"]["ci"], dtype=float)
                table = typical_table(ci, cam["slot"], cam["day"] != held)
                idx = valid_index(cam, h)
                if not len(idx):
                    continue
                k = h * STEPS
                tr, te = idx[cam["day"][idx] != held], idx[cam["day"][idx] == held]
                for part, sel in ((train_parts, tr), (test_parts, te)):
                    if len(sel):
                        part.append((design(cam, table, h, sel, True), ci[sel + k] - ci[sel], ci[sel], ci[sel + k]))
            if not train_parts or not test_parts:
                continue
            A = np.vstack([p[0] for p in train_parts]); y = np.concatenate([p[1] for p in train_parts])
            ok = np.all(np.isfinite(A), axis=1)
            w_full = mf._nonnegative_weights(A[ok].T @ A[ok], A[ok].T @ y[ok], lam=NO_RIDGE)
            w_base = mf._nonnegative_weights(A[ok][:, :4].T @ A[ok][:, :4], A[ok][:, :4].T @ y[ok], lam=NO_RIDGE)
            for X, _, now, act in test_parts:
                good = np.all(np.isfinite(X), axis=1)
                base = np.clip(now[good] + np.clip(X[good][:, :4] @ w_base, -mf.MAX_DELTA, mf.MAX_DELTA), 0, 1)
                full = np.clip(now[good] + np.clip(X[good] @ w_full, -mf.MAX_DELTA, mf.MAX_DELTA), 0, 1)
                e = per_day[held]
                e[0] += float(np.abs(base - act[good]).sum()); e[1] += float(np.abs(full - act[good]).sum()); e[2] += int(good.sum())
        arr = np.array([v for v in per_day.values() if v[2]], dtype=float)
        point = 1.0 - arr[:, 1].sum() / arr[:, 0].sum()
        draws = [1.0 - arr[p][:, 1].sum() / arr[p][:, 0].sum() for p in (rng.integers(0, len(arr), len(arr)) for _ in range(BOOTSTRAPS))]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        result[f"+{h}"] = {"forecasts": int(arr[:, 2].sum()), "gain_vs_shipped_formulas": round(float(point), 4),
                           "ci95": [round(float(lo), 4), round(float(hi), 4)], "adopt": bool(lo > 0)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--min-days", type=int, default=14)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "context_lab.json")
    args = parser.parse_args()
    report = evaluate(read_export(args.csv), args.min_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
