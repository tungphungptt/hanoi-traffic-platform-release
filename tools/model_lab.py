#!/usr/bin/env python3
"""Does a heavier model, relative speed, or calibrated probabilities beat the shipped forecaster?

Leave-one-video-out on the forecast library. A candidate replaces the shipped formula only if its
improvement in MAE over the shipped model has a 95% interval (bootstrap over whole videos) that
stays above zero, at every horizon it would be used for. The rule is fixed here, before looking at
results, so a candidate cannot be chosen because it happened to look good on these few videos.

    python tools/model_lab.py --out artifacts/model_lab.json

Candidates (all predict the change in CI over h minutes from features known at time t):
  gbm          scikit-learn HistGradientBoosting on CI features        (LightGBM is not installed here)
  gbm+speed    the same plus relative speed, occupancy, count, turnover (each minus the camera's own mean so far)
  nnls+speed   the shipped three formulas plus a "speed" formula, non-negative least squares
The GRU of the plan needs PyTorch, which is not installed; it is not tried.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from traffic_brain import minute_forecaster as mf  # noqa: E402

HORIZONS = (5, 10, 15)
STEPS_PER_MIN = int(round(60.0 / mf.GRID_STEP_SECONDS))
BOOTSTRAPS = 2000
SEED = 11


def load_records(library_dir: Path) -> list[dict]:
    records = []
    for path in sorted(library_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.setdefault("source_video_keys", [record["video_key"]])
        series = mf.record_to_series(record)
        if series is not None:
            records.append({"src": record["source_video_keys"][0], "series": series, "name": path.stem})
    return records


def _relative(values: np.ndarray) -> np.ndarray:
    """Value minus the camera's own mean so far (causal): absolute levels differ between roads."""
    return values - mf._running_mean(values)


def feature_table(series: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    base = mf.compute_features(series)
    ci = np.asarray(series["ci"], dtype=float)
    cols = [base[:, 0], base[:, 1], base[:, 2], ci]
    names = ["ci_dev_running", "ci_slope_3m", "ci_dev_10m", "ci"]
    for key in ("motion", "occupancy", "count", "turnover"):
        values = np.asarray(series.get(key, np.full_like(ci, np.nan)), dtype=float)
        rel = _relative(values)
        slope = np.array([mf._slope_per_minute(values, i, 3 * STEPS_PER_MIN, mf.GRID_STEP_SECONDS) for i in range(len(values))])
        cols += [rel, slope]
        names += [f"{key}_rel", f"{key}_slope3m"]
    return np.column_stack(cols), names


def anchors_of(series: dict[str, np.ndarray]) -> np.ndarray:
    """Indices where the shipped model also forecasts (past warm-up, camera on air long enough)."""
    ci = np.asarray(series["ci"], dtype=float)
    t = np.asarray(series["t"], dtype=float)
    observed = np.cumsum(np.isfinite(ci))
    minimum = int((mf.WARMUP_SECONDS - mf.WINDOW_SECONDS) / mf.GRID_STEP_SECONDS) + 1
    base = mf.compute_features(series)
    ok = np.all(np.isfinite(base), axis=1) & (t >= mf.WARMUP_SECONDS) & (observed >= minimum)
    return np.flatnonzero(ok)


def pairs(series: dict[str, np.ndarray], h: int, columns: list[int] | None = None):
    """(X, y_delta, ci_now, ci_actual, index) for every anchor whose target exists."""
    X, names = feature_table(series)
    ci = np.asarray(series["ci"], dtype=float)
    k = h * STEPS_PER_MIN
    idx = anchors_of(series)
    idx = idx[idx + k < len(ci)]
    idx = idx[np.isfinite(ci[idx + k])]
    return X[idx], ci[idx + k] - ci[idx], ci[idx], ci[idx + k], idx, names


def speed_component(Xrow: np.ndarray, names: list[str], h: int) -> np.ndarray:
    """A formula in the style of the shipped ones: slower than the camera's usual -> congestion rises."""
    return -Xrow[:, names.index("motion_rel")] * (1.0 - math.exp(-h / mf.REVERSION_TAU_MINUTES))


def nnls_matrix(X: np.ndarray, names: list[str], h: int, with_speed: bool) -> np.ndarray:
    feats = X[:, [names.index("ci_dev_running"), names.index("ci_slope_3m"), names.index("ci_dev_10m")]]
    cols = [mf._component_matrix(np.column_stack([feats[:, 0], feats[:, 1], feats[:, 2]]), h)]
    if with_speed:
        cols.append(np.nan_to_num(speed_component(X, names, h))[:, None])
    return np.hstack(cols)


def fit_predict(kind: str, h: int, train: list[dict], test: dict) -> np.ndarray | None:
    parts = [pairs(r["series"], h) for r in train]
    Xtr = np.vstack([p[0] for p in parts if len(p[0])])
    ytr = np.concatenate([p[1] for p in parts if len(p[0])])
    Xte, _, _, _, _, names = pairs(test["series"], h)
    if not len(Xte):
        return None
    if kind in ("gbm", "gbm+speed"):
        use = [names.index(n) for n in ("ci_dev_running", "ci_slope_3m", "ci_dev_10m", "ci")] if kind == "gbm" else list(range(len(names)))
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, l2_regularization=5.0,
                                              min_samples_leaf=80, loss="squared_error", random_state=SEED)
        model.fit(Xtr[:, use], ytr)
        return model.predict(Xte[:, use])
    if kind == "nnls+speed":
        A = nnls_matrix(Xtr, names, h, True)
        keep = np.all(np.isfinite(A), axis=1)
        w = mf._nonnegative_weights(A[keep].T @ A[keep], A[keep].T @ ytr[keep])
        return nnls_matrix(Xte, names, h, True) @ w
    raise ValueError(kind)


def shipped_predictions(records: list[dict], index: int) -> dict[int, dict[float, float]]:
    """The deployed model (library of other videos + this video's own past), keyed by horizon and anchor time."""
    own = records[index]["src"]
    library = [r["series"] for j, r in enumerate(records) if j != index and r["src"] != own]
    out: dict[int, dict[float, float]] = {h: {} for h in HORIZONS}
    for anchor in mf.forecast_series(records[index]["series"], library=library):
        for item in anchor["horizons"]:
            out[item["horizon_minutes"]][anchor["t"]] = item["ci"]
    return out


def bootstrap_gain(per_video: dict[str, tuple[float, float, int]], rng: np.random.Generator) -> tuple[float, float, float]:
    """Relative MAE gain of the candidate over the shipped model, 95% interval resampling whole videos.
    per_video[src] = (sum |err| shipped, sum |err| candidate, n)."""
    keys = list(per_video)
    arr = np.array([per_video[k] for k in keys], dtype=float)
    point = 1.0 - arr[:, 1].sum() / arr[:, 0].sum()
    draws = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(keys), len(keys))
        s = arr[pick]
        draws.append(1.0 - s[:, 1].sum() / s[:, 0].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def compare_models(records: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    shipped = [shipped_predictions(records, i) for i in range(len(records))]
    result: dict = {}
    for kind in ("gbm", "gbm+speed", "nnls+speed"):
        result[kind] = {}
        for h in HORIZONS:
            per_video: dict[str, list[float]] = {}
            for i, rec in enumerate(records):
                train = [r for r in records if r["src"] != rec["src"]]
                pred = fit_predict(kind, h, train, rec)
                if pred is None:
                    continue
                _, _, now, act, idx, _ = pairs(rec["series"], h)
                t = np.asarray(rec["series"]["t"], dtype=float)
                cand = np.clip(now + np.clip(pred, -mf.MAX_DELTA, mf.MAX_DELTA), 0.0, 1.0)
                for j, a, c in zip(idx, act, cand):
                    base = shipped[i][h].get(round(float(t[j]), 3))
                    if base is None:
                        continue
                    e = per_video.setdefault(rec["src"], [0.0, 0.0, 0])
                    e[0] += abs(base - a)
                    e[1] += abs(c - a)
                    e[2] += 1
            point, lo, hi = bootstrap_gain({k: tuple(v) for k, v in per_video.items()}, rng)
            n = sum(v[2] for v in per_video.values())
            result[kind][f"+{h}"] = {"forecasts": n, "videos": len(per_video), "gain_vs_shipped": round(point, 4),
                                     "ci95": [round(lo, 4), round(hi, 4)], "adopt": bool(lo > 0)}
    return result


# ------------------------------------------------------------------ probability calibration

def onset_rows(records: list[dict]) -> list[dict]:
    rows = []
    for i, rec in enumerate(records):
        own = rec["src"]
        library = [r["series"] for j, r in enumerate(records) if j != i and r["src"] != own]
        for anchor in mf.forecast_series(rec["series"], library=library):
            for item in anchor["horizons"]:
                if item["actual_ci"] is None or anchor["observed_ci"] >= mf.SEVERE_CI:
                    continue
                rows.append({"src": own, "h": item["horizon_minutes"], "p": float(item["p_gridlock"]),
                             "y": int(item["actual_ci"] >= mf.SEVERE_CI), "now": float(anchor["observed_ci"]),
                             "pred": float(item["ci"])})
    return rows


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration(rows: list[dict], target_precision: float = 0.5) -> dict:
    """Nested: the calibrator and the alert threshold for each held-out video come from the other videos only."""
    out = {}
    for h in HORIZONS:
        pool = [r for r in rows if r["h"] == h]
        if not pool:
            continue
        sources = sorted({r["src"] for r in pool})
        raw_p, cal_p, ys = [], [], []
        tp = fp = pos = 0
        for src in sources:
            train = [r for r in pool if r["src"] != src]
            test = [r for r in pool if r["src"] == src]
            ptr = np.array([r["p"] for r in train])
            ytr = np.array([r["y"] for r in train])
            pte = np.array([r["p"] for r in test])
            yte = np.array([r["y"] for r in test])
            if ytr.sum() < 5:
                cal = pte
                threshold = None
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(ptr, ytr)
                cal = iso.predict(pte)
                trained = iso.predict(ptr)
                threshold = None
                for g in np.arange(0.05, 0.95, 0.01):  # lowest level whose precision on the other videos reaches the target
                    fire = trained >= g
                    if fire.sum() >= 5 and ytr[fire].mean() >= target_precision:
                        threshold = float(g)
                        break
            raw_p.append(pte); cal_p.append(cal); ys.append(yte)
            pos += int(yte.sum())
            if threshold is not None:
                fire = cal >= threshold
                tp += int((fire & (yte == 1)).sum())
                fp += int((fire & (yte == 0)).sum())
        raw, cal, y = np.concatenate(raw_p), np.concatenate(cal_p), np.concatenate(ys)
        out[f"+{h}"] = {"forecasts": int(len(y)), "onsets": int(y.sum()), "base_rate": round(float(y.mean()), 4),
                        "brier_raw": round(brier(raw, y), 5), "brier_calibrated": round(brier(cal, y), 5),
                        "brier_constant": round(brier(np.full_like(y, y.mean(), dtype=float), y), 5),
                        f"alerts_at_precision_{target_precision:.2f}": {"caught": tp, "false": fp, "onsets": pos}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=ROOT / "models" / "forecast_library")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "model_lab.json")
    args = parser.parse_args()
    records = load_records(args.library)
    print(f"{len(records)} series from {len({r['src'] for r in records})} videos")
    report = {"rule": "adopt only if the 95% interval of the MAE gain over the shipped model is above 0",
              "models": compare_models(records), "calibration": calibration(onset_rows(records))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind, per in report["models"].items():
        print(kind, {h: (v["gain_vs_shipped"], v["ci95"], v["adopt"]) for h, v in per.items()})
    for h, v in report["calibration"].items():
        print(h, v)


if __name__ == "__main__":
    main()
