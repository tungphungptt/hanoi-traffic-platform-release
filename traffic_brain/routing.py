"""Alternative driving routes between two points in Hanoi from an OSRM server (OpenStreetMap roads).

Durations are OSRM's network travel times: they do not include live congestion.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
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
                       "geometry": {"type": "LineString", "coordinates": geometry["coordinates"]}})
    if not routes:
        raise RouteError("Không có tuyến nào giữa hai điểm này.")
    return sorted(routes, key=lambda r: r["duration_seconds"])


def route_alternatives(origin: tuple[float, float], destination: tuple[float, float], timeout: float = 10.0) -> list[dict[str, Any]]:
    """Up to three routes, fastest first. ``origin``/``destination`` are (lat, lon)."""
    key = tuple(round(v, 4) for v in (*origin, *destination))
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] <= CACHE_SECONDS:
            return hit[1]
    coords = f"{origin[1]:.6f},{origin[0]:.6f};{destination[1]:.6f},{destination[0]:.6f}"
    query = urlencode({"alternatives": "3", "overview": "full", "geometries": "geojson", "steps": "false"})
    request = Request(f"{_base_url()}/route/v1/driving/{coords}?{query}",
                      headers={"Accept": "application/json", "User-Agent": "hanoiFlow/9 (route alternatives)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read(5_000_000).decode("utf-8"))
    except Exception as exc:
        raise RouteError("Không truy cập được dịch vụ chỉ đường; thử lại sau.") from exc
    routes = parse_routes(payload)
    with _lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = (now, routes)
    return routes
