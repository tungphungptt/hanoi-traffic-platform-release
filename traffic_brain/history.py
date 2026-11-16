"""Durable, timestamped record of what the cameras measured, with the weather at the time.

This is the dataset the forecaster cannot be trained on yet: weeks of 1 Hz measurements with real
clock times, plus a weather snapshot next to them, so that time-of-week and rain effects can be
learned and checked later. SQLite from the standard library, one file (``LIVE_DB``). On hosts
with an ephemeral disk (Render free tier) the file is lost on restart: export it regularly
(``GET /api/live/export``) or mount a persistent disk.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SAMPLE_COLUMNS = ("occupancy_ratio", "vehicle_count", "normalized_speed", "turnover")
WEATHER_COLUMNS = ("temperature_c", "humidity_pct", "precipitation_mm", "wind_kmh", "weather_code")
EXPORT_HEADER = ("camera_id", "t_epoch", "t_local", "ci", *SAMPLE_COLUMNS, "weather_t_epoch", *WEATHER_COLUMNS)
LOCAL_UTC_OFFSET = 7 * 3600  # Hanoi has no daylight saving time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (camera_id TEXT PRIMARY KEY, name TEXT, lat REAL, lng REAL);
CREATE TABLE IF NOT EXISTS samples (
    camera_id TEXT NOT NULL, t REAL NOT NULL, ci REAL NOT NULL,
    occupancy_ratio REAL, vehicle_count REAL, normalized_speed REAL, turnover REAL,
    PRIMARY KEY (camera_id, t)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS weather (
    camera_id TEXT NOT NULL, t REAL NOT NULL,
    temperature_c REAL, humidity_pct REAL, precipitation_mm REAL, wind_kmh REAL, weather_code INTEGER,
    PRIMARY KEY (camera_id, t)) WITHOUT ROWID;
"""


def _local_text(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch + LOCAL_UTC_OFFSET, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class History:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ write

    def save_camera(self, camera_id: str, name: str | None, lat: float | None, lng: float | None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO cameras VALUES (?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET "
                "name=excluded.name, lat=COALESCE(excluded.lat, lat), lng=COALESCE(excluded.lng, lng)",
                (camera_id, name, lat, lng))

    def add_samples(self, camera_id: str, rows: Iterable[Mapping[str, Any]]) -> int:
        data = [(camera_id, float(r["t"]), float(r["raw_ci"]), *(r.get(k) for k in SAMPLE_COLUMNS)) for r in rows]
        if not data:
            return 0
        with self._lock:
            before = self._db.total_changes
            self._db.execute("BEGIN")
            self._db.executemany("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?)", data)
            self._db.execute("COMMIT")
            return self._db.total_changes - before

    def add_weather(self, camera_id: str, t: float, snapshot: Mapping[str, Any]) -> None:
        values = [snapshot.get(k) for k in WEATHER_COLUMNS]
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?)", (camera_id, float(t), *values))

    # ------------------------------------------------------------ read

    def cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT camera_id, name, lat, lng FROM cameras ORDER BY camera_id").fetchall()
        return [{"camera_id": a, "name": b, "lat": c, "lng": d} for a, b, c, d in rows]

    def recent(self, camera_id: str, seconds: float, now: float) -> list[dict[str, Any]]:
        """Samples of the last ``seconds`` before the newest sample (not before ``now``): a restart must not
        throw away history just because the server was down for a while."""
        with self._lock:
            row = self._db.execute("SELECT MAX(t) FROM samples WHERE camera_id=?", (camera_id,)).fetchone()
            if not row or row[0] is None:
                return []
            newest = min(float(row[0]), now + 60)
            found = self._db.execute(
                "SELECT t, ci, occupancy_ratio, vehicle_count, normalized_speed, turnover FROM samples "
                "WHERE camera_id=? AND t>? AND t<=? ORDER BY t", (camera_id, newest - seconds, newest)).fetchall()
        out = []
        for t, ci, *rest in found:
            item = {"t": t, "raw_ci": ci}
            item.update({k: v for k, v in zip(SAMPLE_COLUMNS, rest) if v is not None})
            out.append(item)
        return out

    def counts(self) -> dict[str, Any]:
        with self._lock:
            n, first, last = self._db.execute("SELECT COUNT(*), MIN(t), MAX(t) FROM samples").fetchone()
            w = self._db.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
            cams = self._db.execute("SELECT COUNT(DISTINCT camera_id) FROM samples").fetchone()[0]
        return {"samples": n, "weather_snapshots": w, "cameras": cams, "first_epoch": first, "last_epoch": last,
                "days": round((last - first) / 86400, 2) if n else 0.0}

    def export_rows(self, camera_id: str | None = None, since: float | None = None,
                    until: float | None = None, limit: int = 100_000) -> Iterator[tuple]:
        """Samples joined with the latest weather snapshot of the same camera taken at or before each sample."""
        where, args = [], []
        if camera_id:
            where.append("s.camera_id=?"); args.append(camera_id)
        if since is not None:
            where.append("s.t>=?"); args.append(since)
        if until is not None:
            where.append("s.t<=?"); args.append(until)
        sql = ("SELECT s.camera_id, s.t, s.ci, s.occupancy_ratio, s.vehicle_count, s.normalized_speed, s.turnover, "
               "w.t, w.temperature_c, w.humidity_pct, w.precipitation_mm, w.wind_kmh, w.weather_code "
               "FROM samples s LEFT JOIN weather w ON w.camera_id=s.camera_id AND w.t="
               "(SELECT MAX(t) FROM weather WHERE camera_id=s.camera_id AND t<=s.t)"
               + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY s.camera_id, s.t LIMIT ?")
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        yield from rows

    def export_csv(self, camera_id: str | None = None, since: float | None = None, until: float | None = None,
                   limit: int = 100_000) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(EXPORT_HEADER)
        for cam, t, ci, *rest in self.export_rows(camera_id, since, until, limit):
            writer.writerow([cam, f"{t:.3f}", _local_text(t), ci, *["" if v is None else v for v in rest]])
        return buffer.getvalue()
