from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from traffic_brain.traffic_levels import level_for_ci as _level


def forecast_from_history(history, *, now_epoch=None):
    now = time.time() if now_epoch is None else float(now_epoch)
    rows = []
    for item in history or []:
        try:
            ts = float(item["observed_epoch"])
            ci = float(item["ci"])
            if math.isfinite(ts) and math.isfinite(ci) and 0 <= ci <= 1 and 0 <= now - ts <= 900:
                rows.append({"observed_epoch": ts, "ci": ci, "speed_kmh": item.get("speed_kmh")})
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda row: row["observed_epoch"])
    latest = rows[-1] if rows else None
    base = {
        "available": False,
        "method": "linear_trend_baseline",
        "validated": False,
        "sample_count": len(rows),
        "observed_at": datetime.fromtimestamp(latest["observed_epoch"], timezone.utc).isoformat(timespec="seconds") if latest else None,
        "current_ci": latest["ci"] if latest else None,
        "current_speed_kmh": latest.get("speed_kmh") if latest else None,
        "predictions": [],
    }
    if len(rows) < 5:
        return {**base, "reason": "insufficient_samples", "message": "Cần ít nhất 5 quan sát telemetry thật của cùng đoạn đường."}
    span = rows[-1]["observed_epoch"] - rows[0]["observed_epoch"]
    base["span_seconds"] = round(span, 1)
    if span < 240:
        return {**base, "reason": "short_history", "message": "Chuỗi quan sát cần trải dài ít nhất 4 phút trước khi ngoại suy tham khảo."}
    age = now - rows[-1]["observed_epoch"]
    if age > 180:
        return {**base, "reason": "stale_telemetry", "message": "Telemetry mới nhất đã quá 180 giây."}

    xs = [row["observed_epoch"] - rows[0]["observed_epoch"] for row in rows]
    ys = [row["ci"] for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        return {**base, "reason": "invalid_timestamps", "message": "Không đủ biến thiên thời gian để xây dựng baseline."}
    slope_per_second = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    current = ys[-1]
    predictions = []
    for horizon in (5, 10):
        ci = min(1.0, max(0.0, current + slope_per_second * horizon * 60.0))
        level_key, level_label = _level(ci)
        predictions.append({"horizon_minutes": horizon, "ci": ci, "level_key": level_key, "level": level_label})
    key, label = _level(current)
    return {
        **base,
        "available": True,
        "reason": None,
        "current_level_key": key,
        "current_level": label,
        "predictions": predictions,
        "message": "Baseline ngoại suy tuyến tính từ telemetry thật; chưa phải mô hình AI đã hiệu chuẩn/kiểm định cho Hà Nội.",
    }
