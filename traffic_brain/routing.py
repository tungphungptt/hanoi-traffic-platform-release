"""Alternative driving routes between two points in Hanoi from an OSRM server (OpenStreetMap roads).

Durations are OSRM's network travel times: they do not include live congestion.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_OSRM = "https://router.project-osrm.org"
CACHE_SECONDS = 600
_cache: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}
_lock = threading.Lock()


class RouteError(RuntimeError):
    pass


def _base_url() -> str:
    base = os.getenv("ROUTING_API_BASE_URL", DEFAULT_OSRM).strip().rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://localhost") or base.startswith("http://127.0.0.1")):
        raise RouteError("ROUTING_API_BASE_URL phải là HTTPS (HTTP chỉ cho localhost).")
    return base


def parse_routes(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("code") != "Ok" or not isinstance(payload.get("routes"), list):
        raise RouteError("Dịch vụ chỉ đường không trả về tuyến hợp lệ.")
    routes = []
    for route in payload["routes"][:3]:
        geometry = route.get("geometry") or {}
        if not isinstance(geometry, dict) or len(geometry.get("coordinates") or []) < 2:
            continue
        routes.append({"distance_m": round(float(route.get("distance", 0)), 1),
                       "duration_seconds": round(float(route.get("duration", 0)), 1),
                       "search_method": "osrm_alternative",
                       "geometry": {"type": "LineString", "coordinates": geometry["coordinates"]}})
    if not routes:
        raise RouteError("Không có tuyến nào giữa hai điểm này.")
    return sorted(routes, key=lambda r: r["duration_seconds"])


def _request_routes(points: list[tuple[float, float]], timeout: float, alternatives: str) -> list[dict[str, Any]]:
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    query = urlencode({"alternatives": alternatives, "overview": "full", "geometries": "geojson", "steps": "false"})
    request = Request(f"{_base_url()}/route/v1/driving/{coords}?{query}",
                      headers={"Accept": "application/json", "User-Agent": "hanoiFlow/9 (route alternatives)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read(5_000_000).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise RouteError("Không truy cập được dịch vụ chỉ đường; thử lại sau.") from exc
    return parse_routes(payload)


def _detour_waypoints(
    origin: tuple[float, float],
    destination: tuple[float, float],
    waypoint_is_valid: Callable[[float, float], bool] | None,
) -> list[tuple[float, float]]:
    lat = (origin[0] + destination[0]) / 2
    lon = (origin[1] + destination[1]) / 2
    x_scale = 111_320 * math.cos(math.radians(lat))
    dx = (destination[1] - origin[1]) * x_scale
    dy = (destination[0] - origin[0]) * 110_540
    length = math.hypot(dx, dy)
    if length < 1_500 or x_scale <= 0:
        return []

    offset = min(2_500, max(700, length * 0.15))
    perpendicular_x, perpendicular_y = -dy / length, dx / length
    waypoints = []
    for side in (-1, 1):
        candidate = (
            lat + perpendicular_y * offset * side / 110_540,
            lon + perpendicular_x * offset * side / x_scale,
        )
        if waypoint_is_valid is None or waypoint_is_valid(*candidate):
            waypoints.append(candidate)
    return waypoints


def _route_distance_to_point(route: dict[str, Any], point: tuple[float, float]) -> float:
    latitude, longitude = point
    x_scale = 111_320 * math.cos(math.radians(latitude))
    best = math.inf
    coordinates = route["geometry"]["coordinates"]
    for first, second in zip(coordinates, coordinates[1:]):
        ax, ay = (first[0] - longitude) * x_scale, (first[1] - latitude) * 110_540
        bx, by = (second[0] - longitude) * x_scale, (second[1] - latitude) * 110_540
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        t = 0 if denominator == 0 else max(0, min(1, -(ax * dx + ay * dy) / denominator))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best


def route_alternatives(
    origin: tuple[float, float],
    destination: tuple[float, float],
    timeout: float = 10.0,
    waypoint_is_valid: Callable[[float, float], bool] | None = None,
) -> list[dict[str, Any]]:
    """Find up to three distinct routes, fastest first, supplementing OSRM alternatives with via-points."""
    key = tuple(round(v, 4) for v in (*origin, *destination))
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] <= CACHE_SECONDS:
            return hit[1]

    routes = _request_routes([origin, destination], timeout, "3")
    if len(routes) < 3:
        for waypoint in _detour_waypoints(origin, destination, waypoint_is_valid):
            if len(routes) >= 3:
                break
            if any(_route_distance_to_point(route, waypoint) <= 300 for route in routes):
                continue
            try:
                candidates = _request_routes([origin, waypoint, destination], min(timeout, 5.0), "false")
            except RouteError:
                continue
            candidate = next((route for route in candidates
                              if _route_distance_to_point(route, waypoint) <= 300), None)
            if candidate is not None:
                candidate["search_method"] = "via_waypoint"
                routes.append(candidate)

    routes = sorted(routes[:3], key=lambda route: route["duration_seconds"])
    with _lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = (now, routes)
    return routes
