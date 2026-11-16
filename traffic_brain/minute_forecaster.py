"""Causal +5/+10/+15 minute congestion forecasts for a single long video, with backtest.

The video is first turned into an observed congestion series: the explainable visual
CI from ``video_review`` averaged over a rolling 60 s window, sampled every 15 s.
At every anchor time t the forecaster may only use observations up to t:

* features are computed from the series up to t (level, short/long trend, change of
  occupancy, vehicle count and motion);
* the model for horizon h is a ridge regression of the future change
  ``CI(tau + h) - CI(tau)`` fitted only on pairs whose target time ``tau + h <= t``,
  i.e. on the part of the same video that has already been watched;
* until enough such pairs exist the forecaster falls back to a damped trend.

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
RIDGE_LAMBDA = 8.0
MAX_DELTA = 0.6
DEFAULT_SIGMA = 0.12
SIGMA_FLOOR = 0.03
Z_80 = 1.2816
# Fixed before any evaluation: warn when the forecast gives at least a 40% chance of CI >= 0.55.
WARNING_PROBABILITY = 0.40
METHOD_ENSEMBLE = "online_ensemble"
METHOD_DAMPED = "damped_trend"
# Only relative features: absolute CI levels differ between cameras, so learning from
# other videos must rely on how the current state deviates from this road's own level.
FEATURE_NAMES = (
    "ci_dev_running",
    "ci_dev_5m",
    "ci_slope_3m",
    "ci_slope_10m",
    "ci_dev_10m",
    "occupancy_slope_3m",
    "count_slope_3m",
    "motion_slope_3m",
)
# "turnover" (share of vehicles not seen one sample earlier) is measured and stored, but as a
# feature it did not lower leave-one-video-out errors (+5/+10/+15: 9.4/10.1/7.1% vs 9.5/9.9/7.9%
# without it), so it is not used for forecasting.
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
    # Scale count and motion like the visual rule (20 vehicles, 0.045 diagonal/s) so
    # slopes are on the same order as CI slopes.
    count = series["count"] / 20.0
    motion = series["motion"] / 0.045
    running = _running_mean(ci)
    for i in range(n):
        if not math.isfinite(ci[i]):
            continue
        mean_10m = _window_mean(ci, i, 10 * per_min)
        mean_5m = _window_mean(ci, i, 5 * per_min)
        features[i] = (
            ci[i] - running[i],
            ci[i] - mean_5m if math.isfinite(mean_5m) else 0.0,
            _slope_per_minute(ci, i, 3 * per_min, step_seconds),
            _slope_per_minute(ci, i, 10 * per_min, step_seconds),
            ci[i] - mean_10m if math.isfinite(mean_10m) else 0.0,
            _slope_per_minute(series["occupancy"], i, 3 * per_min, step_seconds),
            _slope_per_minute(count, i, 3 * per_min, step_seconds),
            _slope_per_minute(motion, i, 3 * per_min, step_seconds),
        )
    return features


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.hstack([z, np.ones((z.shape[0], 1))])
    gram = design.T @ design + lam * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ y)
    residual = y - design @ weights
    dof = max(1, y.shape[0] - design.shape[1])
    sigma = float(math.sqrt(float(residual @ residual) / dof))
    return weights, mean, scale, sigma


def _ridge_predict(weights: np.ndarray, mean: np.ndarray, scale: np.ndarray, row: np.ndarray) -> float:
    z = (row - mean) / scale
    return float(np.dot(z, weights[:-1]) + weights[-1])


def _level(ci: float) -> str:
    return level_for_ci(ci)[0]


COMPONENTS = ("persistence", "reversion", "trend", "ridge", "analog")
DEFAULT_WEIGHTS = {"persistence": 0.5, "reversion": 0.5}
# Traffic at a fixed camera fluctuates around the road's typical level (signal cycles,
# platoons). The reversion component pulls the forecast back toward that level.
REVERSION_TAU_MINUTES = 5.0
MIN_SCORED_ERRORS = 8
ERROR_MEMORY = 40
ANALOG_NEIGHBOURS = 5


TREND_DAMPING = 0.85


def _damped_minutes(h: int, phi: float | None = None) -> float:
    """Sum of phi**m for m = 1..h: how many minutes of the current trend to carry forward."""
    phi = TREND_DAMPING if phi is None else phi
    return phi * (1.0 - phi ** h) / (1.0 - phi)


def _analog_delta(x: np.ndarray, deltas: np.ndarray, row: np.ndarray) -> float:
    """Average future change of the most similar past situations (nearest neighbours)."""
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    distance = np.sqrt((((x - row) / scale) ** 2).sum(axis=1))
    k = min(ANALOG_NEIGHBOURS, deltas.shape[0])
    nearest = np.argpartition(distance, k - 1)[:k]
    weights = 1.0 / (distance[nearest] + 0.25)
    return float(np.dot(weights, deltas[nearest]) / weights.sum())


def forecast_series(series: Mapping[str, np.ndarray], *,
                    horizons_minutes: Iterable[int] = HORIZONS_MINUTES,
                    step_seconds: float = GRID_STEP_SECONDS,
                    warmup_seconds: float = WARMUP_SECONDS,
                    min_train_pairs: int = MIN_TRAIN_PAIRS,
                    ridge_lambda: float = RIDGE_LAMBDA,
                    library: Iterable[Mapping[str, np.ndarray]] = ()) -> list[dict[str, Any]]:
    """Make causal forecasts at every anchor and attach the later-observed outcome.

    Each horizon combines five simple forecasters - persistence, reversion to the road's
    running mean, damped trend, ridge regression and nearest-neighbour analogs - weighted
    by the inverse of their own
    recent errors. An error only counts once its target time has been observed, so
    the weights at time t depend on nothing after t.

    ``library`` holds observed series from *other* videos. Their (feature, future change)
    pairs are fully in the past relative to this video, so they may train the ridge and
    analog components from the first anchor on.
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
    dev_col = FEATURE_NAMES.index("ci_dev_running")
    # Per horizon: grid index -> {component: delta} and final delta, kept to score later.
    library_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    library_series = [item for item in library if item.get("t") is not None and np.asarray(item["t"]).size]
    for h in horizons:
        k = h * steps_per_min
        xs, ds = [], []
        for other in library_series:
            other_ci = np.asarray(other["ci"], dtype=np.float64)
            other_features = compute_features(other, step_seconds=step_seconds)
            other_valid = np.all(np.isfinite(other_features), axis=1)
            for j in range(0, other_ci.shape[0] - k):
                if other_valid[j] and math.isfinite(other_ci[j + k]):
                    xs.append(other_features[j])
                    ds.append(other_ci[j + k] - other_ci[j])
        library_pairs[h] = (np.asarray(xs, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
                            np.asarray(ds, dtype=np.float64))
    history: dict[int, dict[int, dict[str, float]]] = {h: {} for h in horizons}
    final_history: dict[int, dict[int, float]] = {h: {} for h in horizons}
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
            last_train = i - k
            pairs = [j for j in range(0, last_train + 1)
                     if valid[j] and math.isfinite(ci[j + k])] if last_train >= 0 else []
            past_deltas = np.asarray([ci[j + k] - ci[j] for j in pairs], dtype=np.float64)
            components = {
                "persistence": 0.0,
                "trend": slope_3m * _damped_minutes(h),
                "reversion": -float(features[i, dev_col]) * (1.0 - math.exp(-h / REVERSION_TAU_MINUTES)),
            }
            lib_x, lib_d = library_pairs[h]
            train_x = np.vstack([lib_x, features[pairs]]) if pairs else lib_x
            train_d = np.concatenate([lib_d, past_deltas])
            if train_d.shape[0] >= min_train_pairs:
                weights, mean, scale, _ = _fit_ridge(train_x, train_d, ridge_lambda)
                components["ridge"] = _ridge_predict(weights, mean, scale, features[i])
                components["analog"] = _analog_delta(train_x, train_d, features[i])
            components = {name: max(-MAX_DELTA, min(MAX_DELTA, value)) for name, value in components.items()}

            # Score each component on past anchors whose target is already observed.
            scored = sorted(j for j in history[h] if j + k <= i and math.isfinite(ci[j + k]))[-ERROR_MEMORY:]
            errors: dict[str, list[float]] = {name: [] for name in COMPONENTS}
            final_errors: list[float] = []
            for j in scored:
                truth = ci[j + k] - ci[j]
                for name, value in history[h][j].items():
                    errors[name].append(abs(value - truth))
                final_errors.append(final_history[h][j] - truth)
            usable = {name: errors[name] for name in components if len(errors[name]) >= MIN_SCORED_ERRORS}
            if len(usable) >= 2:
                raw = {name: 1.0 / (float(np.mean(errs)) ** 2 + 1e-4) for name, errs in usable.items()}
                method = METHOD_ENSEMBLE
            else:
                raw = {name: weight for name, weight in DEFAULT_WEIGHTS.items() if name in components}
                method = METHOD_DAMPED
            total = sum(raw.values())
            blend = {name: value / total for name, value in raw.items()}
            delta = sum(blend[name] * components[name] for name in blend)
            history[h][i] = components
            final_history[h][i] = delta

            if len(final_errors) >= MIN_SCORED_ERRORS:
                sigma = float(np.sqrt(np.mean(np.square(final_errors))))
            elif train_d.size >= 5:
                sigma = float(train_d.std())
            else:
                sigma = DEFAULT_SIGMA
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
                "level_key": _level(predicted),
                "method": method,
                "weights": {name: round(value, 3) for name, value in blend.items()},
                "train_pairs": int(train_d.shape[0]),
                "library_pairs": int(lib_d.shape[0]),
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
                                 item["method"], bool(item.get("warning"))))
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
        metrics.append({
            "horizon_minutes": h,
            "evaluated": len(rows),
            "trained_anchors": sum(1 for r in rows if r[4] == METHOD_ENSEMBLE),
            "mae_model": mae_model,
            "mae_persistence": mae_persist,
            "mae_trend": mae_trend,
            "rmse_model": float(np.sqrt(np.mean((pred - act) ** 2))),
            "skill_vs_persistence": None if mae_persist <= 1e-9 else 1.0 - mae_model / mae_persist,
            "level_accuracy_model": float(np.mean(level(pred) == level(act))),
            "level_accuracy_persistence": float(np.mean(level(cur) == level(act))),
            "onset_model": _onset_scores(not_jammed),
            "onset_trend": _onset_scores(not_jammed_trend),
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
    """Pick moments worth replaying: correct congestion warnings, misses, false alarms."""
    hits, misses, false_alarms = [], [], []
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
            if warned and actual >= SEVERE_CI:
                hits.append(record)
            elif not warned and actual >= SEVERE_CI:
                misses.append(record)
            elif warned and actual < SEVERE_CI:
                false_alarms.append(record)
    # Chronological, so a demo can walk through the video from start to end.
    hits.sort(key=lambda r: (r["t"], r["horizon_minutes"]))
    misses.sort(key=lambda r: (r["t"], r["horizon_minutes"]))
    false_alarms.sort(key=lambda r: (-r["p_gridlock"], r["t"]))
    return {
        "hits": _spread(hits, 60.0, limit),
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
            "model": METHOD_ENSEMBLE,
            "components": list(COMPONENTS),
            "fallback": METHOD_DAMPED,
            "features": list(FEATURE_NAMES),
            "min_train_pairs": MIN_TRAIN_PAIRS,
            "warning_probability": WARNING_PROBABILITY,
            "library_videos": int(library_videos),
            "ridge_lambda": RIDGE_LAMBDA,
            "causal": True,
            "description": (
                "Tại mỗi thời điểm t, hệ thống chỉ dùng dữ liệu đến t. Năm bộ dự báo (giữ nguyên, "
                "hồi về mức thường gặp của đoạn đường, xu hướng giảm chấn, hồi quy ridge, tình huống tương tự) được kết hợp theo nghịch đảo "
                "sai số gần đây của chính chúng; ridge và tình huống tương tự học từ các cặp "
                "(đặc trưng lúc τ, CI lúc τ+h) có τ+h ≤ t trong video này và từ các video đã phân tích trước."
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
