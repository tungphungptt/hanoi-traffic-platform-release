from __future__ import annotations

"""Camera-independent feature mapping primitives for the Hanoi traffic proposal.

This module intentionally does not pretend that detector boxes alone yield calibrated
km/h or ground-truth road occupancy. Speed requires a per-camera homography and
correct timestamps; output quality includes every fallback feature used.
"""

import math
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence


class FeatureContractError(ValueError):
    """The observation, calibration, or manifest contract is incomplete or invalid."""


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(f"{field} phải là số hữu hạn") from exc
    if not math.isfinite(number):
        raise FeatureContractError(f"{field} phải là số hữu hạn")
    return number


def project_pixel_to_ground(point_px: Sequence[float], homography: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Project a pixel point (normally vehicle bbox bottom-center) to ground-plane metres."""
    if len(point_px) != 2 or len(homography) != 3 or any(len(row) != 3 for row in homography):
        raise FeatureContractError("Cần điểm ảnh (x,y) và ma trận homography 3x3")
    x = _finite_number(point_px[0], "pixel_x")
    y = _finite_number(point_px[1], "pixel_y")
    h = [[_finite_number(v, "homography") for v in row] for row in homography]
    determinant = (
        h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
    )
    if abs(determinant) < 1e-10:
        raise FeatureContractError("Ma trận homography suy biến; kiểm tra lại điểm hiệu chuẩn")
    denominator = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denominator) < 1e-9:
        raise FeatureContractError("Điểm ảnh ánh xạ về vô cực; kiểm tra homography/ROI")
    ground_x = (h[0][0] * x + h[0][1] * y + h[0][2]) / denominator
    ground_y = (h[1][0] * x + h[1][1] * y + h[1][2]) / denominator
    if not math.isfinite(ground_x) or not math.isfinite(ground_y):
        raise FeatureContractError("Homography tạo tọa độ mặt đường không hữu hạn")
    return ground_x, ground_y


def estimate_speed_kmh(
    previous_pixel: Sequence[float],
    current_pixel: Sequence[float],
    elapsed_seconds: float,
    homography: Sequence[Sequence[float]],
    *,
    max_speed_kmh: float = 180.0,
) -> float | None:
    """Estimate speed from two tracked points on a calibrated ground plane; reject implausible samples."""
    dt = _finite_number(elapsed_seconds, "elapsed_seconds")
    if dt <= 0 or dt > 120:
        return None
    previous = project_pixel_to_ground(previous_pixel, homography)
    current = project_pixel_to_ground(current_pixel, homography)
    distance_m = math.hypot(current[0] - previous[0], current[1] - previous[1])
    speed = distance_m / dt * 3.6
    if not math.isfinite(speed) or speed < 0 or speed > max_speed_kmh:
        return None
    return round(speed, 4)


def aggregate_traffic_features(
    observations: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    previous_features: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build manifest-ordered features from tracked vehicle observations.

    Each observation has ``track_id``, ``class_id``, and optionally calibrated
    ``speed_kmh`` / ``active_now``. Provide one entry per track observation in the
    configured rolling window; flow counts/PCU are normalized to per-minute units,
    while linear density and area occupancy use active objects only. ``active_now`` marks objects used for current occupancy/speed;
    if omitted, the latest row per track is used. This is an adapter primitive, not
    a complete RTSP/ByteTrack worker.
    """
    detector = manifest.get("detector_config", {})
    mapping = detector.get("class_mapping", {})
    calibration = manifest.get("camera_calibration", {})
    schema = manifest.get("feature_schema", {})
    ordered_keys = schema.get("ordered_input_keys", [])
    fallbacks = schema.get("default_fallback_values", {})
    if not isinstance(mapping, Mapping) or not isinstance(ordered_keys, list) or not ordered_keys:
        raise FeatureContractError("Manifest thiếu detector_config.class_mapping hoặc ordered_input_keys")
    road_length_m = _finite_number(calibration.get("effective_road_length_m", 0), "effective_road_length_m")
    road_area_m2 = _finite_number(calibration.get("effective_road_area_m2", 0), "effective_road_area_m2")
    window_seconds = _finite_number(schema.get("window_seconds", 60), "feature_schema.window_seconds")
    if road_length_m <= 0 or road_area_m2 <= 0:
        raise FeatureContractError("Cần hiệu chuẩn effective_road_length_m và effective_road_area_m lớn hơn 0")
    if window_seconds <= 0:
        raise FeatureContractError("feature_schema.window_seconds phải lớn hơn 0")

    all_rows = [dict(row) for row in observations]
    # Keep the last observation of each track in the supplied window.
    latest_by_track: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for idx, row in enumerate(all_rows):
        track_id = str(row.get("track_id", "")).strip()
        if not track_id:
            raise FeatureContractError(f"Observation #{idx} thiếu track_id")
        class_id = str(row.get("class_id", ""))
        if class_id not in mapping:
            continue
        latest_by_track[track_id] = row

    current_rows = [row for row in latest_by_track.values() if row.get("active_now") is not False]
    counts: dict[str, int] = {
        str(meta.get("name", "")).strip().lower(): 0
        for meta in mapping.values()
        if isinstance(meta, Mapping) and str(meta.get("name", "")).strip()
    }
    pcu_total = 0.0
    occupied_area = 0.0
    speeds: list[float] = []
    for class_id, row in ((str(row.get("class_id")), row) for row in latest_by_track.values()):
        meta = mapping[class_id]
        class_name = str(meta.get("name", "")).strip().lower()
        if not class_name:
            continue
        counts[class_name] = counts.get(class_name, 0) + 1
        pcu_total += _finite_number(meta.get("pcu_weight", 1.0), f"pcu_weight.{class_name}")

    for row in current_rows:
        meta = mapping[str(row.get("class_id"))]
        area = meta.get("avg_area_m2")
        if area is not None:
            occupied_area += max(0.0, _finite_number(area, "avg_area_m2"))
        raw_speed = row.get("speed_kmh")
        if raw_speed is not None:
            speed = _finite_number(raw_speed, "speed_kmh")
            if 0 <= speed <= 180:
                speeds.append(speed)

    feature_values: dict[str, float] = {}
    for class_name, count in counts.items():
        feature_values[f"flow.{class_name}_count"] = float(count)
    # Normalize singular/plural class names where the trained schema uses a common alias.
    aliases = {"automobile": "car", "truck": "truck", "bus": "bus", "motorbike": "motorcycle", "motorcycle": "motorcycle"}
    for class_name, count in counts.items():
        alias = aliases.get(class_name)
        if alias:
            feature_values[f"flow.{alias}_count"] = float(count)
    # The proposal defines flow features per minute; normalize arbitrary configured windows.
    per_minute = 60.0 / window_seconds
    for class_name, count in list(counts.items()):
        feature_values[f"flow.{class_name}_count"] = float(count) * per_minute
        alias = aliases.get(class_name)
        if alias:
            feature_values[f"flow.{alias}_count"] = float(count) * per_minute
    feature_values["flow.pcu_equivalent"] = float(pcu_total) * per_minute
    active_count = sum(1 for row in current_rows if str(row.get("class_id", "")) in mapping)
    feature_values["density.linear_veh_km"] = float(active_count / road_length_m * 1000.0)
    feature_values["density.areal_occupancy"] = min(1.0, occupied_area / road_area_m2)

    previous = previous_features or {}
    quality_fallbacks: list[str] = []
    if speeds:
        mean_speed = sum(speeds) / len(speeds)
        feature_values["speed.mean_kmh"] = mean_speed
        if len(speeds) >= 2:
            variance = sum((speed - mean_speed) ** 2 for speed in speeds) / len(speeds)
            std_speed = math.sqrt(variance)
            feature_values["speed.std_kmh"] = std_speed
            feature_values["speed.cv_ratio"] = std_speed / max(mean_speed, 1e-6)
        # CV with fewer than two calibrated samples remains missing and is explicitly flagged as a fallback below.
    if "flow.pcu_equivalent" in previous:
        feature_values["temporal.pcu_lag_1"] = _finite_number(previous["flow.pcu_equivalent"], "previous.pcu")
    if "speed.mean_kmh" in feature_values and "speed.mean_kmh" in previous:
        feature_values["temporal.speed_delta_1"] = feature_values["speed.mean_kmh"] - _finite_number(previous["speed.mean_kmh"], "previous.speed")

    result: dict[str, float] = {}
    for key in ordered_keys:
        if not isinstance(key, str):
            raise FeatureContractError("Mọi phần tử ordered_input_keys phải là chuỗi")
        if key in feature_values:
            result[key] = float(feature_values[key])
        elif key in fallbacks:
            result[key] = _finite_number(fallbacks[key], f"fallback.{key}")
            quality_fallbacks.append(key)
        else:
            raise FeatureContractError(f"Thiếu đặc trưng '{key}' và không có fallback khai báo trong manifest")
    return {
        "features": result,
        "quality": {
            "track_count_in_window": len(latest_by_track),
            "active_track_count": len(current_rows),
            "calibrated_speed_samples": len(speeds),
            "used_fallback_keys": quality_fallbacks,
            "ready_for_real_forecast": not quality_fallbacks,
            "warning": None if not quality_fallbacks else "Một số đặc trưng dùng fallback của manifest; không coi là đo đạc thật.",
        },
    }
