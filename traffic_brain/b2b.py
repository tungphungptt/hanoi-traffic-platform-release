"""Compact per-camera summary of the saved demo analyses, for the business (B2B) console."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SERIES_POINTS = 48
_cache: dict[str, Any] = {"stamp": None, "data": None}


def _downsample(observed: list[dict[str, Any]], points: int = SERIES_POINTS) -> list[float]:
    values = [round(float(o["ci"]), 3) for o in observed if o.get("ci") is not None]
    if len(values) <= points:
        return values
    step = len(values) / points
    return [values[min(len(values) - 1, int(i * step))] for i in range(points)]


def _camera_row(label: str, mf: dict[str, Any]) -> dict[str, Any] | None:
    latest = mf.get("latest")
    series = _downsample(mf.get("observed") or [])
    if not latest or not series:
        return None
    skill = {m["horizon_minutes"]: m.get("skill_vs_persistence") for m in mf.get("metrics", []) if m.get("evaluated")}
    horizons = [{"minutes": h["horizon_minutes"], "ci": round(h["ci"], 3), "level": h["level_key"],
                 "p_gridlock": round(h["p_gridlock"], 3), "warning": bool(h.get("warning")), "watch": bool(h.get("watch")),
                 "skill": None if skill.get(h["horizon_minutes"]) is None else round(skill[h["horizon_minutes"]], 3)}
                for h in latest.get("horizons", [])]
    return {"camera": label, "now_ci": round(latest["observed_ci"], 3), "level": latest["level_key"],
            "peak_ci": max(series), "mean_ci": round(sum(series) / len(series), 3), "series": series, "horizons": horizons}


def _entry_rows(entry_id: str, label: str, session: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cameras = session.get("cameras") or []
    for cam in cameras:
        mf = cam.get("minute_forecast") or {}
        name = label if len(cameras) == 1 else f"{cam.get('label', 'Camera')} (bản ghi 2 camera)"
        row = _camera_row(name, mf)
        if row:
            row["id"] = f"{entry_id}:{cam.get('id', 0)}"
            row["demo_id"] = entry_id
            rows.append(row)
    if not rows:
        row = _camera_row(label, session.get("minute_forecast") or {})
        if row:
            row["id"] = f"{entry_id}:0"
            row["demo_id"] = entry_id
            rows.append(row)
    return rows


def overview(artifacts: Path) -> dict[str, Any]:
    """Summary of every configured demo; cached until a file changes."""
    config_path = artifacts / "demo.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = [("main", config.get("label", config["title"]), config["session_file"])]
    entries += [(e["id"], e.get("label", e["title"]), e["session_file"]) for e in config.get("extras", [])]
    stamp = tuple((p.name, p.stat().st_mtime_ns) for p in sorted([config_path, *(artifacts / f for _, _, f in entries)]))
    if _cache["stamp"] == stamp:
        return _cache["data"]
    cameras: list[dict[str, Any]] = []
    for entry_id, label, session_file in entries:
        session = json.loads((artifacts / session_file).read_text(encoding="utf-8"))
        cameras += _entry_rows(entry_id, label, session)
    warnings = sum(1 for c in cameras for h in c["horizons"] if h["warning"])
    data = {"cameras": cameras,
            "totals": {"cameras": len(cameras), "gridlock_now": sum(1 for c in cameras if c["level"] == "gridlock"),
                       "busy_now": sum(1 for c in cameras if c["level"] == "busy"),
                       "warnings": warnings,
                       "mean_ci": round(sum(c["now_ci"] for c in cameras) / len(cameras), 3) if cameras else None},
            "note": "Số liệu lấy từ các bản ghi camera đã phân tích sẵn, không phải truyền hình trực tiếp."}
    _cache.update(stamp=stamp, data=data)
    return data
