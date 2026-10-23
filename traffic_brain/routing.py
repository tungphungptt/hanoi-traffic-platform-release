"""Road-network route alternatives via OSRM plus conservative live-telemetry scoring."""
from __future__ import annotations
import json, math, os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any

class RouteServiceError(RuntimeError):
    pass

def _distance_point_to_segment_m(point: tuple[float,float], a: tuple[float,float], b: tuple[float,float]) -> float:
    # Equirectangular approximation is sufficient for local camera-to-route proximity in Hanoi.
    lat0 = math.radians(point[1])
    sx = 111_320.0 * math.cos(lat0); sy = 110_540.0
    px, py = point[0]*sx, point[1]*sy
    ax, ay = a[0]*sx, a[1]*sy; bx, by = b[0]*sx, b[1]*sy
    dx, dy = bx-ax, by-ay
    den = dx*dx+dy*dy
    t = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/den)) if den else 0.0
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))

def route_has_nearby_live_telemetry(geometry: list[list[float]], telemetry: list[dict[str,Any]], *, radius_m: float=350.0, max_age_seconds: float=180.0, now_epoch: float | None=None) -> list[dict[str,Any]]:
    import time
    now = time.time() if now_epoch is None else now_epoch
    valid = []
    for item in telemetry:
        if item.get("source") != "rtsp":
            continue
        try:
            age = now - float(item["observed_epoch"])
            lat = float(item["lat"]); lon = float(item["lng"]); ci = float(item["ci"])
        except (KeyError, TypeError, ValueError):
            continue
        if age < 0 or age > max_age_seconds or not (math.isfinite(ci) and 0 <= ci <= 1):
            continue
        point = (lon, lat)
        if len(geometry) == 1:
            distance = math.hypot((lon-geometry[0][0])*111320*math.cos(math.radians(lat)), (lat-geometry[0][1])*110540)
        else:
            distance = min(_distance_point_to_segment_m(point, tuple(geometry[i]), tuple(geometry[i+1])) for i in range(len(geometry)-1))
        if distance <= radius_m:
            valid.append({**item, "distance_to_route_m": round(distance, 1), "age_seconds": round(age, 1)})
    return valid

def score_routes(routes: list[dict[str,Any]], telemetry: list[dict[str,Any]], *, now_epoch: float | None=None) -> dict[str,Any]:
    if not routes:
        raise RouteServiceError("Dịch vụ chỉ đường không trả về tuyến nào.")
    scored = []
    for i, route in enumerate(routes):
        geometry_obj = route.get("geometry") or {}
        coords = geometry_obj.get("coordinates", []) if isinstance(geometry_obj, dict) else []
        nearby = route_has_nearby_live_telemetry(coords, telemetry, now_epoch=now_epoch)
        ci_values = [float(x["ci"]) for x in nearby]
        scored.append({
            "route_id": f"R{i+1}", "distance_m": round(float(route.get("distance", 0)), 1),
            "duration_seconds": round(float(route.get("duration", 0)), 1),
            "geometry": {"type":"LineString", "coordinates": coords},
            "nearby_live_cameras": [{k:v for k,v in x.items() if k not in {"source_url","credentials"}} for x in nearby],
            "live_camera_count": len(nearby), "mean_live_ci": round(sum(ci_values)/len(ci_values), 3) if ci_values else None,
            "traffic_status": "live_telemetry_available" if nearby else "no_live_telemetry",
        })
    all_covered = len(scored) > 1 and all(r["live_camera_count"] > 0 for r in scored)
    if all_covered:
        for route in scored:
            # Route ETA from OSRM is a static network estimate; congestion risk adds a
            # conservative penalty from fresh RTSP measurements near that route.
            route["comparison_score"] = round(route["duration_seconds"] * (1.0 + 0.35*route["mean_live_ci"]), 1)
        best = min(scored, key=lambda r:r["comparison_score"])
        basis = "fresh_live_camera_telemetry_and_osrm_base_time"
        note = "Các phương án đều có camera RTSP mới gần tuyến. Điểm so sánh là heuristic vận hành, không thay thế dự báo đã kiểm định."
    else:
        best = min(scored, key=lambda r:r["duration_seconds"])
        basis = "osrm_reference_duration_only"
        note = "Chưa có telemetry RTSP mới bao phủ mọi phương án. Tuyến được ưu tiên chỉ theo thời gian đường tham chiếu, chưa thể khẳng định ít tắc hơn."
    for r in scored:
        r["recommended"] = r["route_id"] == best["route_id"]
        r["comparison_score"] = r.get("comparison_score")
    return {"routes": scored, "recommended_route_id": best["route_id"], "recommendation_basis": basis,
            "traffic_aware": all_covered, "message": note}

def fetch_route_alternatives(origin: dict[str,Any], destination: dict[str,Any], *, base_url: str | None=None, timeout_seconds: float=8.0) -> list[dict[str,Any]]:
    base = (base_url or os.getenv("ROUTING_API_BASE_URL", "https://router.project-osrm.org")).strip().rstrip("/")
    if not base.startswith("https://") and not base.startswith("http://localhost") and not base.startswith("http://127.0.0.1"):
        raise RouteServiceError("ROUTING_API_BASE_URL phải là HTTPS (HTTP chỉ cho localhost).")
    try:
        coords = f"{float(origin['lng']):.6f},{float(origin['lat']):.6f};{float(destination['lng']):.6f},{float(destination['lat']):.6f}"
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteServiceError("Thiếu tọa độ đầu/cuối hợp lệ.") from exc
    query = urlencode({"alternatives":"3", "overview":"full", "geometries":"geojson", "steps":"false"})
    url = f"{base}/route/v1/driving/{coords}?{query}"
    request = Request(url, headers={"Accept":"application/json", "User-Agent":"HanoiTrafficPlatform/6.0 (local route alternative demo)"})
    try:
        with urlopen(request, timeout=max(2.0, min(float(timeout_seconds), 20.0))) as response:
            raw = response.read(3_000_001)
            if len(raw) > 3_000_000:
                raise RouteServiceError("Phản hồi chỉ đường lớn hơn 3 MB.")
            payload = json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        raise RouteServiceError(f"Dịch vụ chỉ đường trả HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RouteServiceError(f"Không truy cập được dịch vụ chỉ đường ({type(exc).__name__}).") from exc
    if not isinstance(payload, dict) or payload.get("code") != "Ok" or not isinstance(payload.get("routes"), list):
        raise RouteServiceError("Dịch vụ chỉ đường không trả route hợp lệ.")
    result = []
    for route in payload["routes"][:3]:
        geometry = route.get("geometry", {})
        coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        if len(coords) < 2:
            continue
        result.append({"distance": route.get("distance"), "duration": route.get("duration"), "geometry": geometry})
    if not result:
        raise RouteServiceError("Không có geometry đường đi trong phản hồi.")
    return result
