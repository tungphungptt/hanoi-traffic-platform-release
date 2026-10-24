from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "geo" / "hanoi_boundary.geojson"
HANOI_CENTER = (21.0285, 105.8542)
BOUNDARY_SOURCE = "OpenStreetMap administrative boundary relation 1903516 (ODbL)"

@lru_cache(maxsize=1)
def _boundary_geometries() -> tuple[dict[str, Any], ...]:
    if not BOUNDARY_PATH.is_file():
        raise RuntimeError(f"Hanoi boundary file missing: {BOUNDARY_PATH}")
    with BOUNDARY_PATH.open("r", encoding="utf-8-sig") as stream:
        doc = json.load(stream)
    features = doc.get("features", []) if doc.get("type") == "FeatureCollection" else [doc]
    geometries = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        if geometry.get("type") in ("Polygon", "MultiPolygon") and geometry.get("coordinates"):
            geometries.append(geometry)
    if not geometries:
        raise RuntimeError("Hanoi boundary contains no Polygon/MultiPolygon geometry")
    return tuple(geometries)

def _point_on_segment(x: float, y: float, a: list, b: list, eps: float = 1e-10) -> bool:
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps

def _ring_contains(x: float, y: float, ring: list) -> bool:
    if not isinstance(ring, list) or len(ring) < 4:
        return False
    inside = False
    previous = ring[-1]
    for current in ring:
        if _point_on_segment(x, y, previous, current):
            return True
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross_x:
                inside = not inside
        previous = current
    return inside

def _polygon_contains(x: float, y: float, rings: list) -> bool:
    if not rings or not _ring_contains(x, y, rings[0]):
        return False
    # GeoJSON holes must exclude points inside their inner rings.
    return not any(_ring_contains(x, y, hole) for hole in rings[1:])

def is_hanoi_coordinate(latitude: Any, longitude: Any) -> bool:
    """Return True only for coordinates inside the bundled Hanoi boundary polygon."""
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    x, y = lon, lat
    for geometry in _boundary_geometries():
        kind = geometry.get("type")
        coords = geometry.get("coordinates") or []
        polygons = [coords] if kind == "Polygon" else coords
        for polygon in polygons:
            if _polygon_contains(x, y, polygon):
                return True
    return False
