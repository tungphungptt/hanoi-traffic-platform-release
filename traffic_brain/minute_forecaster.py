"""Causal +5/+10/+15 minute congestion forecasts for a single long video, with backtest.

The video is first turned into an observed congestion series: the explainable visual
CI from ``video_review`` averaged over a rolling 60 s window, sampled every 15 s.
At every anchor time t the forecaster may only use observations up to t:

* the forecast for horizon h is CI(t) plus a weighted sum of three fixed formulas of the
  series up to t: pull toward the camera's mean so far, pull toward its last 10 minutes,
  and the damped 3-minute trend;
* the weights are non-negative least squares on pairs ``(formulas at tau, CI(tau + h) -
  CI(tau))`` from other videos and from this video with ``tau + h <= t``;
* until enough such pairs exist fixed default weights are used.

Because every forecast is made before its target is seen, the rest of the video can be
used to check it. The backtest compares the model with persistence (no change) and a
plain linear trend, the two simplest honest baselines. The ground truth is the same
visual CI, not an expert label, and a single video is not a benchmark.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from traffic_brain.traffic_levels import BUSY_CI, SEVERE_CI, STATE_LABELS, level_for_ci

GRID_STEP_SECONDS = 15.0
WINDOW_SECONDS = 60.0
WARMUP_SECONDS = 180.0
HORIZONS_MINUTES = (5, 10, 15)
MIN_TRAIN_PAIRS = 12
MAX_DELTA = 0.6
DEFAULT_SIGMA = 0.12
SIGMA_FLOOR = 0.03
Z_80 = 1.2816
# Fixed before any evaluation: warn when the forecast gives at least a 40% chance of CI >= 0.55.
WARNING_PROBABILITY = 0.40
# Softer "watch" flag for 10-15 min ahead, where 40% is almost never reached. Set after the
# leave-one-video-out check of 2026-11-16 (precision there about 30-40% at +10/+15 against a
# ~7% base rate), so treat those numbers as optimistic until checked on new videos.
WATCH_PROBABILITY = 0.20
# Relative features only: absolute CI levels differ between cameras, so learning from other
# videos must rely on how the current state deviates from this road's own level. Occupancy,
# count, motion and turnover slopes were tried as ridge-regression inputs and added nothing
# over these three (see models/forecast_library/README.md); they are stored, not used.
FEATURE_NAMES = (
    "ci_dev_running",
    "ci_slope_3m",
    "ci_dev_10m",
)
SERIES_KEYS = ("t", "ci", "occupancy", "count", "motion", "turnover")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_observed_series(timeline: Iterable[Mapping[str, Any]], *,
                          step_seconds: float = GRID_STEP_SECONDS,
                          window_seconds: float = WINDOW_SECONDS) -> dict[str, np.ndarray]:
    """Average 1 Hz frame rows into a rolling-window series on a fixed grid.

    Uses ``raw_ci`` (falls back to ``ci``), ``occupancy_ratio``, ``vehicle_count``,
    ``normalized_speed`` and ``turnover`` (share of vehicles not seen in the previous sample).
    A grid point with too few samples in its window is NaN.
    """
    rows = []
    for row in timeline or []:
        if not isinstance(row, Mapping):
            continue
        t = _finite(row.get("timestamp_seconds"))
        ci = _finite(row.get("raw_ci", row.get("ci")))
        if t is None or ci is None:
            continue
        occ = _finite(row.get("occupancy_ratio"))
        count = _finite(row.get("vehicle_count"))
        speed = _finite(row.get("normalized_speed"))
        turnover = _finite(row.get("turnover"))
        rows.append((t, min(1.0, max(0.0, ci)),
                     np.nan if occ is None else occ,
                     np.nan if count is None else count,
                     np.nan if speed is None else speed,
                     np.nan if turnover is None else turnover))
    empty = {key: np.zeros(0) for key in SERIES_KEYS}
    if not rows:
        return empty
    rows.sort(key=lambda item: item[0])
    data = np.asarray(rows, dtype=np.float64)
    times = data[:, 0]
    if times[-1] < window_seconds:
        return empty
    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    sample_period = float(np.median(deltas)) if deltas.size else 1.0
    min_samples = max(3, int(0.5 * window_seconds / max(sample_period, 1e-6)))
    grid = np.arange(window_seconds, times[-1] + 1e-9, step_seconds)
    out = {key: np.full(grid.shape, np.nan) for key in SERIES_KEYS[1:]}
    lo = hi = 0
    for i, tau in enumerate(grid):
        while hi < len(times) and times[hi] <= tau + 1e-9:
            hi += 1
        while lo < hi and times[lo] <= tau - window_seconds + 1e-9:
            lo += 1
        if hi - lo < min_samples:
            continue
        chunk = data[lo:hi]
        out["ci"][i] = float(np.mean(chunk[:, 1]))
        for key, col in (("occupancy", 2), ("count", 3), ("motion", 4), ("turnover", 5)):
            values = chunk[:, col]
            values = values[np.isfinite(values)]
            if values.size:
                out[key][i] = float(np.median(values)) if key == "motion" else float(np.mean(values))
    out["t"] = grid
    return out


def _slope_per_minute(values: np.ndarray, end: int, steps: int, step_seconds: float) -> float:
    start = max(0, end - steps)
    y = values[start:end + 1]
    x = np.arange(start, end + 1, dtype=np.float64) * step_seconds / 60.0
    mask = np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x = x[mask] - x[mask].mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y[mask] - y[mask].mean()) / denom)


def _window_mean(values: np.ndarray, end: int, steps: int) -> float:
    chunk = values[max(0, end - steps):end + 1]
    chunk = chunk[np.isfinite(chunk)]
    return float(chunk.mean()) if chunk.size else float("nan")


def _running_mean(values: np.ndarray) -> np.ndarray:
    """Mean of all finite values up to and including each index (causal)."""
    finite = np.isfinite(values)
    sums = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def compute_features(series: Mapping[str, np.ndarray], *,
                     step_seconds: float = GRID_STEP_SECONDS) -> np.ndarray:
    """Return one causal feature row per grid point (uses indices <= i only)."""
    ci = series["ci"]
    n = ci.shape[0]
    per_min = int(round(60.0 / step_seconds))
    features = np.full((n, len(FEATURE_NAMES)), np.nan)
    running = _running_mean(ci)
    for i in range(n):
        if not math.isfinite(ci[i]):
            continue
        mean_10m = _window_mean(ci, i, 10 * per_min)
        features[i] = (
            ci[i] - running[i],
            _slope_per_minute(ci, i, 3 * per_min, step_seconds),
            ci[i] - mean_10m if math.isfinite(mean_10m) else 0.0,
        )
    return features


def _level(ci: float) -> str:
    return level_for_ci(ci)[0]


# The forecast for horizon h is CI(t) + sum_c w_c * component_c. Every component is a fixed
# formula of the series up to t (nothing is fitted inside it), so their weights can be
# learned by least squares on (component values, observed change) pairs from other videos
# and from the already-watched part of this video, at every anchor, at little cost.
COMPONENTS = ("reversion", "recent_mean", "trend")
# Traffic at a fixed camera fluctuates around the road's typical level (signal cycles,
# platoons). "reversion" pulls toward the camera's mean so far, "recent_mean" toward the
# last 10 minutes, "trend" carries the 3-minute slope forward (damped).
REVERSION_TAU_MINUTES = 5.0
TREND_DAMPING = 0.85
# Used only while fewer than MIN_TRAIN_PAIRS pairs exist (no library, start of a video).
# Least-squares weights on all 22 shipped camera series (2026-11-16).
DEFAULT_WEIGHTS = {
    5: {"reversion": 0.42, "recent_mean": 0.70, "trend": 0.04},
    10: {"reversion": 0.31, "recent_mean": 0.53, "trend": 0.0},
    15: {"reversion": 0.38, "recent_mean": 0.32, "trend": 0.0},
}
WEIGHT_RIDGE = 1.0  # small pull of the weights toward 0 (= keep the current level)
MIN_SCORED_ERRORS = 8
ERROR_MEMORY = 40
METHOD_BLEND = "level_blend"
METHOD_DEFAULT = "default_weights"


def _damped_minutes(h: int, phi: float | None = None) -> float:
    """Sum of phi**m for m = 1..h: how many minutes of the current trend to carry forward."""
    phi = TREND_DAMPING if phi is None else phi
    return phi * (1.0 - phi ** h) / (1.0 - phi)


def _component_matrix(features: np.ndarray, h: int) -> np.ndarray:
    """Rows of (reversion, recent_mean, trend) predicted changes for horizon h."""
    dev_running = features[:, FEATURE_NAMES.index("ci_dev_running")]
    dev_10m = features[:, FEATURE_NAMES.index("ci_dev_10m")]
    slope = features[:, FEATURE_NAMES.index("ci_slope_3m")]
    return np.column_stack([
        -dev_running * (1.0 - math.exp(-h / REVERSION_TAU_MINUTES)),
        -dev_10m,
        slope * _damped_minutes(h),
    ])


def _nonnegative_weights(gram: np.ndarray, rhs: np.ndarray, lam: float = WEIGHT_RIDGE) -> np.ndarray:
    """argmin ||y - A w||^2 + lam ||w||^2 with w >= 0, from A'A and A'y (all subsets)."""
    m = gram.shape[0]
    best, best_loss = np.zeros(m), 0.0
    for mask in range(1, 2 ** m):
        idx = [c for c in range(m) if mask >> c & 1]
        sub = gram[np.ix_(idx, idx)] + lam * np.eye(len(idx))
        try:
            w = np.linalg.solve(sub, rhs[idx])
        except np.linalg.LinAlgError:
            continue
        if np.any(w < 0):
            continue
        loss = float(w @ sub @ w - 2.0 * w @ rhs[idx])
        if loss < best_loss:
            best = np.zeros(m)
            best[idx] = w
            best_loss = loss
    return best


def forecast_series(series: Mapping[str, np.ndarray], *,
                    horizons_minutes: Iterable[int] = HORIZONS_MINUTES,
                    step_seconds: float = GRID_STEP_SECONDS,
                    warmup_seconds: float = WARMUP_SECONDS,
                    min_train_pairs: int = MIN_TRAIN_PAIRS,
                    library: Iterable[Mapping[str, np.ndarray]] = ()) -> list[dict[str, Any]]:
    """Make causal forecasts at every anchor and attach the later-observed outcome.

    ``library`` holds observed series from *other* videos; their pairs are entirely in the
    past relative to this video. Pairs from this video join once their target time has
    been observed, so the forecast at time t depends on nothing after t.
    """
    times = series["t"]
    ci = series["ci"]
    if times.size == 0:
        return []
    features = compute_features(series, step_seconds=step_seconds)
    valid = np.all(np.isfinite(features), axis=1)
    horizons = [int(h) for h in horizons_minutes]
    steps_per_min = int(round(60.0 / step_seconds))
    slope_col = FEATURE_NAMES.index("ci_slope_3m")
    library_series = [item for item in library if item.get("t") is not None and np.asarray(item["t"]).size]
    library_features = [compute_features(other, step_seconds=step_seconds) for other in library_series]
    # Sufficient statistics per horizon: A'A, A'y, y'y, n for the library and this video.
    stats: dict[int, dict[str, Any]] = {}
    for h in horizons:
        k = h * steps_per_min
        gram, rhs, yy, n = np.zeros((3, 3)), np.zeros(3), 0.0, 0
        for other, other_features in zip(library_series, library_features):
            other_ci = np.asarray(other["ci"], dtype=np.float64)
            if other_ci.shape[0] <= k:
                continue
            a = _component_matrix(other_features[:-k], h)
            y = other_ci[k:] - other_ci[:-k]
            keep = np.all(np.isfinite(a), axis=1) & np.isfinite(y)
            a, y = a[keep], y[keep]
            gram += a.T @ a
            rhs += a.T @ y
            yy += float(y @ y)
            n += int(y.shape[0])
        stats[h] = {"gram": gram, "rhs": rhs, "yy": yy, "n": n, "library_n": n, "next": 0,
                    "components": _component_matrix(features, h)}
    history: dict[int, dict[int, float]] = {h: {} for h in horizons}
    anchors: list[dict[str, Any]] = []
    # A camera that comes on air mid-video (broadcast recordings) needs its own warm-up.
    observed_so_far = np.cumsum(np.isfinite(ci))
    min_observed = int((warmup_seconds - WINDOW_SECONDS) / step_seconds) + 1
    for i, tau in enumerate(times):
        if tau < warmup_seconds or not valid[i] or observed_so_far[i] < min_observed:
            continue
        ci_now = float(ci[i])
        slope_3m = float(features[i, slope_col])
        entry: dict[str, Any] = {
            "t": round(float(tau), 3),
            "observed_ci": ci_now,
            "level_key": _level(ci_now),
            "horizons": [],
        }
        for h in horizons:
            k = h * steps_per_min
            s = stats[h]
            comps = s["components"]
            # Add this video's pairs whose target time is now observed (j + k <= i).
            while s["next"] + k <= i:
                j = s["next"]
                s["next"] += 1
                if valid[j] and math.isfinite(ci[j + k]):
                    a, y = comps[j], ci[j + k] - ci[j]
                    s["gram"] += np.outer(a, a)
                    s["rhs"] += a * y
                    s["yy"] += y * y
                    s["n"] += 1
            if s["n"] >= min_train_pairs:
                w = _nonnegative_weights(s["gram"], s["rhs"])
                method = METHOD_BLEND
                resid = (s["yy"] - 2.0 * float(w @ s["rhs"]) + float(w @ s["gram"] @ w)) / s["n"]
                fit_sigma = math.sqrt(max(resid, 0.0))
            else:
                w = np.asarray([DEFAULT_WEIGHTS.get(h, DEFAULT_WEIGHTS[15])[c] for c in COMPONENTS])
                method = METHOD_DEFAULT
                fit_sigma = DEFAULT_SIGMA
            delta = max(-MAX_DELTA, min(MAX_DELTA, float(w @ comps[i])))
            history[h][i] = delta
            # Spread from this video's own scored errors once there are enough of them.
            scored = sorted(j for j in history[h] if j + k <= i and math.isfinite(ci[j + k]))[-ERROR_MEMORY:]
            final_errors = [history[h][j] - (ci[j + k] - ci[j]) for j in scored]
            if len(final_errors) >= MIN_SCORED_ERRORS:
                sigma = float(np.sqrt(np.mean(np.square(final_errors))))
            else:
                sigma = fit_sigma
            sigma = max(SIGMA_FLOOR, sigma)
            predicted = min(1.0, max(0.0, ci_now + delta))
            trend = min(1.0, max(0.0, ci_now + slope_3m * h))
            target_index = i + k
            actual = float(ci[target_index]) if target_index < ci.shape[0] and math.isfinite(ci[target_index]) else None
            p_gridlock = 1.0 - _normal_cdf((SEVERE_CI - predicted) / sigma)
            entry["horizons"].append({
                "horizon_minutes": h,
                "target_t": round(float(tau + h * 60.0), 3),
                "ci": predicted,
                "lower": max(0.0, predicted - Z_80 * sigma),
                "upper": min(1.0, predicted + Z_80 * sigma),
                "p_gridlock": p_gridlock,
                "warning": p_gridlock >= WARNING_PROBABILITY,
                "watch": WATCH_PROBABILITY <= p_gridlock < WARNING_PROBABILITY,
                "level_key": _level(predicted),
                "method": method,
                "weights": {name: round(float(v), 3) for name, v in zip(COMPONENTS, w)},
                "train_pairs": int(s["n"]),
                "library_pairs": int(s["library_n"]),
                "persistence_ci": ci_now,
                "trend_ci": trend,
                "actual_ci": actual,
                "actual_level_key": None if actual is None else _level(actual),
            })
        anchors.append(entry)
    return anchors


def _onset_scores(rows: list[tuple[bool, float]]) -> dict[str, Any]:
    """rows = (warned, actual CI) for anchors that were not yet gridlocked."""
    tp = sum(1 for w, a in rows if w and a >= SEVERE_CI)
    fp = sum(1 for w, a in rows if w and a < SEVERE_CI)
    fn = sum(1 for w, a in rows if not w and a >= SEVERE_CI)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "precision": None if tp + fp == 0 else tp / (tp + fp),
    }


def evaluate_anchors(anchors: list[dict[str, Any]],
                     horizons_minutes: Iterable[int] = HORIZONS_MINUTES) -> list[dict[str, Any]]:
    """Backtest metrics per horizon over anchors whose target was later observed."""
    metrics = []
    for h in horizons_minutes:
        rows = []
        for anchor in anchors:
            for item in anchor["horizons"]:
                if item["horizon_minutes"] == h and item["actual_ci"] is not None:
                    rows.append((anchor["observed_ci"], item["ci"], item["trend_ci"], item["actual_ci"],
                                 item["method"], bool(item.get("warning")),
                                 item.get("p_gridlock", 0.0) >= WATCH_PROBABILITY))
        if not rows:
            metrics.append({"horizon_minutes": h, "evaluated": 0})
            continue
        cur = np.asarray([r[0] for r in rows])
        pred = np.asarray([r[1] for r in rows])
        trend = np.asarray([r[2] for r in rows])
        act = np.asarray([r[3] for r in rows])
        mae_model = float(np.mean(np.abs(pred - act)))
        mae_persist = float(np.mean(np.abs(cur - act)))
        mae_trend = float(np.mean(np.abs(trend - act)))
        level = np.vectorize(_level)
        not_jammed = [(r[5], r[3]) for r in rows if r[0] < SEVERE_CI]
        not_jammed_trend = [(r[2] >= SEVERE_CI, r[3]) for r in rows if r[0] < SEVERE_CI]
        not_jammed_watch = [(r[6], r[3]) for r in rows if r[0] < SEVERE_CI]
        metrics.append({
            "horizon_minutes": h,
            "evaluated": len(rows),
            "trained_anchors": sum(1 for r in rows if r[4] == METHOD_BLEND),
            "mae_model": mae_model,
            "mae_persistence": mae_persist,
            "mae_trend": mae_trend,
            "rmse_model": float(np.sqrt(np.mean((pred - act) ** 2))),
            "skill_vs_persistence": None if mae_persist <= 1e-9 else 1.0 - mae_model / mae_persist,
            "level_accuracy_model": float(np.mean(level(pred) == level(act))),
            "level_accuracy_persistence": float(np.mean(level(cur) == level(act))),
            "onset_model": _onset_scores(not_jammed),
            "onset_trend": _onset_scores(not_jammed_trend),
            "onset_watch": _onset_scores(not_jammed_watch),
            "actual_onsets": sum(1 for _, a in not_jammed if a >= SEVERE_CI),
        })
    return metrics


def _spread(items: list[dict[str, Any]], min_gap_seconds: float, limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for item in items:
        if all(abs(item["t"] - other["t"]) >= min_gap_seconds for other in chosen):
            chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def select_highlights(anchors: list[dict[str, Any]], *, limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    """Pick moments worth replaying: correct warnings, correct watch flags, misses, false alarms."""
    hits, watch_hits, misses, false_alarms = [], [], [], []
    for anchor in anchors:
        if anchor["observed_ci"] >= SEVERE_CI:
            continue
        for item in anchor["horizons"]:
            actual = item["actual_ci"]
            if actual is None:
                continue
            record = {"t": anchor["t"], "horizon_minutes": item["horizon_minutes"],
                      "target_t": item["target_t"], "observed_ci": anchor["observed_ci"],
                      "predicted_ci": item["ci"], "actual_ci": actual,
                      "p_gridlock": item["p_gridlock"]}
            warned = bool(item.get("warning"))
            watched = bool(item.get("watch"))
            if warned and actual >= SEVERE_CI:
                hits.append(record)
            elif watched and actual >= SEVERE_CI:
                watch_hits.append(record)
            elif not warned and actual >= SEVERE_CI:
                misses.append(record)
            elif warned and actual < SEVERE_CI:
                false_alarms.append(record)
    # Chronological, so a demo can walk through the video from start to end.
    hits.sort(key=lambda r: (r["t"], r["horizon_minutes"]))
    watch_hits.sort(key=lambda r: (r["t"], r["horizon_minutes"]))
    misses.sort(key=lambda r: (r["t"], r["horizon_minutes"]))
    false_alarms.sort(key=lambda r: (-r["p_gridlock"], r["t"]))
    return {
        "hits": _spread(hits, 60.0, limit),
        "watch_hits": _spread(watch_hits, 60.0, 3),
        "misses": _spread(misses, 60.0, 3),
        "false_alarms": _spread(false_alarms, 60.0, 3),
    }


def video_key(path: Any) -> str | None:
    """Content fingerprint (size + first/last 4 MiB) so a video never trains on itself."""
    import hashlib
    from pathlib import Path
    try:
        source = Path(path)
        size = source.stat().st_size
        digest = hashlib.sha256(str(size).encode())
        with source.open("rb") as stream:
            digest.update(stream.read(4 * 1024 * 1024))
            if size > 8 * 1024 * 1024:
                stream.seek(size - 4 * 1024 * 1024)
                digest.update(stream.read(4 * 1024 * 1024))
        return digest.hexdigest()[:32]
    except OSError:
        return None


def series_record(series: Mapping[str, np.ndarray], *, video_key: str, filename: str | None = None) -> dict[str, Any]:
    """JSON-safe copy of an observed series for the cross-video training library."""
    def clean(values: np.ndarray) -> list[float | None]:
        return [None if not math.isfinite(float(v)) else round(float(v), 6) for v in values]
    return {
        "contract": "hanoiflow-observed-series/1.0",
        "video_key": video_key,
        "filename": filename,
        "step_seconds": GRID_STEP_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        **{key: clean(np.asarray(series[key], dtype=np.float64)) for key in SERIES_KEYS if key in series},
    }


def record_to_series(record: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """Validate a library record; return None if it does not match this grid."""
    try:
        if record.get("contract") != "hanoiflow-observed-series/1.0":
            return None
        if float(record.get("step_seconds")) != GRID_STEP_SECONDS or float(record.get("window_seconds")) != WINDOW_SECONDS:
            return None
        series = {key: np.asarray([np.nan if v is None else float(v) for v in record[key]], dtype=np.float64)
                  for key in SERIES_KEYS if key in record or key != "turnover"}
    except (KeyError, TypeError, ValueError):
        return None
    sizes = {values.shape[0] for values in series.values()}
    if len(sizes) != 1 or not sizes.pop():
        return None
    return series


def build_minute_forecast(timeline: Iterable[Mapping[str, Any]], *, duration_seconds: float,
                          timing_confirmed: bool,
                          library: Iterable[Mapping[str, np.ndarray]] = (),
                          library_videos: int = 0,
                          series: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Full result for the UI: observed series, causal forecasts, backtest, highlights.

    ``series`` may be passed instead of ``timeline`` when the observed series is already
    known (for example a stored library record).
    """
    series = build_observed_series(timeline) if series is None else series
    library = list(library)
    needed = WARMUP_SECONDS + HORIZONS_MINUTES[0] * 60.0 + 4 * GRID_STEP_SECONDS
    base: dict[str, Any] = {
        "contract": "hanoiflow-minute-forecast/1.0",
        "horizons_minutes": list(HORIZONS_MINUTES),
        "step_seconds": GRID_STEP_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "thresholds": {"busy": BUSY_CI, "gridlock": SEVERE_CI},
        "state_labels": dict(STATE_LABELS),
        "timing_confirmed": bool(timing_confirmed),
        "observed": [
            {"t": round(float(t), 3), "ci": None if not math.isfinite(c) else float(c)}
            for t, c in zip(series["t"], series["ci"])
        ],
        "ground_truth": "visual_ci_60s_mean",
        "method": {
            "model": METHOD_BLEND,
            "components": list(COMPONENTS),
            "fallback": METHOD_DEFAULT,
            "min_train_pairs": MIN_TRAIN_PAIRS,
            "warning_probability": WARNING_PROBABILITY,
            "watch_probability": WATCH_PROBABILITY,
            "library_videos": int(library_videos),
            "causal": True,
            "description": (
                "Tại mỗi thời điểm t, hệ thống chỉ dùng dữ liệu đến t. Mức CI sau h phút = mức hiện tại "
                "cộng tổ hợp của ba xu hướng: kéo về mức trung bình của camera, kéo về mức trung bình "
                "10 phút gần nhất và xu hướng 3 phút (giảm dần). Trọng số học bằng bình phương tối thiểu "
                "trên các video đã phân tích trước và phần video này đã xem xong (τ+h ≤ t)."
            ),
        },
    }
    if duration_seconds < needed or series["t"].size == 0:
        return {**base, "available": False, "anchors": [], "metrics": [], "highlights": {},
                "latest": None, "reason": "video_too_short",
                "message": (f"Cần video dài ít nhất {math.ceil(needed / 60)} phút để dự báo +5 phút và "
                            "khoảng 25–30 phút để kiểm chứng đủ cả +15 phút.")}
    anchors = forecast_series(series, library=library)
    if not anchors:
        return {**base, "available": False, "anchors": [], "metrics": [], "highlights": {},
                "latest": None, "reason": "insufficient_observations",
                "message": "Không đủ khung hình hợp lệ để tạo chuỗi CI quan sát."}
    metrics = evaluate_anchors(anchors)
    verifiable = [m["horizon_minutes"] for m in metrics if m.get("evaluated", 0) > 0]
    message = ("Dự báo được tạo tuần tự, chỉ dùng phần video trước thời điểm dự báo; "
               "phần sau của video dùng để kiểm chứng.")
    if not timing_confirmed:
        message += (" Chưa xác nhận clip quay theo thời gian thực: '5 phút' ở đây là 5 phút trên "
                    "dòng thời gian video.")
    return {
        **base,
        "available": True,
        "anchors": anchors,
        "metrics": metrics,
        "verifiable_horizons": verifiable,
        "highlights": select_highlights(anchors),
        "latest": anchors[-1],
        "message": message,
        "limitations": [
            "Giá trị thực tế dùng để kiểm chứng là CI thị giác (trung bình 60 giây) do chính hệ thống đo, không phải nhãn chuyên gia.",
            "Mô hình học từ phần đầu của cùng video và các video đã phân tích trước; một video không phải benchmark và không chứng minh khả năng tổng quát sang camera khác.",
            "So sánh công bằng nhất là với persistence (giữ nguyên mức hiện tại): mô hình chỉ có đóng góp khi sai số thấp hơn persistence.",
        ],
    }
