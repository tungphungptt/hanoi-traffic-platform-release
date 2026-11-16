"""hanoiFlow web app: upload a traffic video, get +5/+10/+15 minute congestion forecasts.

One CPU worker analyses one video at a time. Results stay in memory for an hour; the
browser plays the user's own file locally and only fetches the analysis result.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import weather
from app.hanoi_scope import is_hanoi_coordinate
from traffic_brain import routing
from traffic_brain.live import LiveError, LiveStore
from traffic_brain.detector import Detector
from traffic_brain.minute_forecaster import record_to_series, series_record, video_key
from traffic_brain.video_analysis import VideoAnalysisError, analyze_video, capture_evidence, result_payload

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
shutil.rmtree(EVIDENCE_DIR, ignore_errors=True)  # leftovers of jobs from before a restart
BUNDLED_LIBRARY_DIR = ROOT / "models" / "forecast_library"
LOCAL_LIBRARY_DIR = DATA_DIR / "forecast_library"
logger = logging.getLogger("hanoiflow")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MIB", 2048, 1, 16384) * 1024 * 1024
MAX_VIDEO_SECONDS = _env_int("MAX_VIDEO_MINUTES", 90, 1, 240) * 60
SAMPLE_FPS = 1.0
SUPPORTED_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".mpeg", ".mpg", ".webm"}
# Shipped training series (models/forecast_library) and series of videos analysed on this machine.
BUNDLED_LIBRARY_ENABLED = _env_flag("FORECAST_BUNDLED_LIBRARY", True)
LOCAL_LIBRARY_ENABLED = _env_flag("FORECAST_LIBRARY_ENABLED", True)
LOCAL_LIBRARY_MAX = 50
JOB_RETENTION_SECONDS = 3600
# Optional shared password for uploads on a public deployment (set UPLOAD_TOKEN on Render).
UPLOAD_TOKEN = os.getenv("UPLOAD_TOKEN", "").strip()
LIVE_TOKEN = os.getenv("LIVE_TOKEN", "").strip()  # empty: live intake is switched off
LIVE_BODY_LIMIT = 256 * 1024
UPLOADS_PER_HOUR = _env_int("UPLOADS_PER_HOUR", 20, 1, 1000)
_upload_times: list[float] = []
DISK_RESERVE_BYTES = 64 * 1024 * 1024

detector = Detector()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
LOCK = threading.RLock()
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="hanoiFlow", version="9.0.0",
              description="Dự báo ùn tắc +5/+10/+15 phút từ video giao thông, kiểm chứng trên chính video.")


class RequestGuard:
    """Reject a bad request before Starlette parses (and spools) its body.

    * ``POST /api/analyze``: upload token (X-Upload-Token), declared size, hourly limit, busy server.
    * ``POST /api/live/<id>/samples``: live token (X-Live-Token) and a small body limit.

    Only headers are read for the checks; a body larger than allowed (also without Content-Length)
    is cut off once it passes the limit.
    """

    def __init__(self, asgi_app):
        self.app = asgi_app

    @staticmethod
    def _check(path: str, headers: dict[str, str]) -> tuple[int, str | None, tuple[int, str] | None]:
        """Returns (body limit, unit label, problem)."""
        if path == "/api/analyze":
            limit = MAX_UPLOAD_BYTES + 1024 * 1024  # multipart overhead
            too_big = (413, f"Video vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.")
            if UPLOAD_TOKEN and not secrets.compare_digest(headers.get("x-upload-token", "").strip(), UPLOAD_TOKEN):
                return limit, too_big[1], (401, "Sai mã tải lên. Hỏi người quản trị máy chủ để lấy mã.")
            if headers.get("content-length", "").isdigit() and int(headers["content-length"]) > limit:
                return limit, too_big[1], too_big
            with LOCK:
                now = _now()
                _upload_times[:] = [t for t in _upload_times if now - t < 3600]
                if _busy():
                    return limit, too_big[1], (409, "Máy chủ đang phân tích một video khác. Vui lòng thử lại sau.")
                if len(_upload_times) >= UPLOADS_PER_HOUR:
                    return limit, too_big[1], (429, "Đã đạt giới hạn số lượt tải video trong giờ này. Vui lòng thử lại sau.")
                _upload_times.append(now)
            return limit, too_big[1], None
        too_big = (413, "Gói số liệu quá lớn.")
        if not LIVE_TOKEN:
            return LIVE_BODY_LIMIT, too_big[1], (503, "Máy chủ chưa bật nhận số liệu trực tiếp (LIVE_TOKEN).")
        if not secrets.compare_digest(headers.get("x-live-token", "").strip(), LIVE_TOKEN):
            return LIVE_BODY_LIMIT, too_big[1], (401, "Sai mã thu nhận trực tiếp.")
        if headers.get("content-length", "").isdigit() and int(headers["content-length"]) > LIVE_BODY_LIMIT:
            return LIVE_BODY_LIMIT, too_big[1], too_big
        return LIVE_BODY_LIMIT, too_big[1], None

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        guarded = scope["type"] == "http" and scope["method"] in ("POST", "DELETE") and (
            (scope["method"] == "POST" and path == "/api/analyze")
            or (path.startswith("/api/live/") and (path.endswith("/samples") or scope["method"] == "DELETE")))
        if not guarded:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        limit, too_big_message, problem = self._check(path, headers)
        if problem:
            body = json.dumps({"detail": problem[1]}).encode()
            await send({"type": "http.response.start", "status": problem[0],
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()), (b"connection", b"close")]})
            return await send({"type": "http.response.body", "body": body})
        received, cut = 0, False

        async def limited_receive():
            nonlocal received, cut
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    cut = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if cut and message["type"] == "http.response.start":
                body = json.dumps({"detail": too_big_message}).encode()
                await send({"type": "http.response.start", "status": 413,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                return await send({"type": "http.response.body", "body": body})
            if cut:
                return None
            return await send(message)

        await self.app(scope, limited_receive, guarded_send)


app.add_middleware(RequestGuard)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------- training library

def _library_files() -> list[Path]:
    files: list[Path] = []
    if BUNDLED_LIBRARY_ENABLED and BUNDLED_LIBRARY_DIR.is_dir():
        files += sorted(BUNDLED_LIBRARY_DIR.glob("*.json"))
    if LOCAL_LIBRARY_ENABLED and LOCAL_LIBRARY_DIR.is_dir():
        files += sorted(LOCAL_LIBRARY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:LOCAL_LIBRARY_MAX]
    return files


def load_library(exclude_key: str | None) -> tuple[list[dict[str, np.ndarray]], int]:
    """Series of other videos. A record never comes from the video being analysed."""
    series, sources = [], set()
    for path in _library_files():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        keys = {record.get("video_key"), *(record.get("source_video_keys") or [])}
        if exclude_key and exclude_key in keys:
            continue
        parsed = record_to_series(record)
        if parsed is not None:
            series.append(parsed)
            sources.add(tuple(sorted(k for k in record.get("source_video_keys") or [record.get("video_key")] if k)))
    return series, len(sources)


def save_to_library(key: str | None, filename: str, cameras: list[dict[str, Any]]) -> None:
    if not LOCAL_LIBRARY_ENABLED or not key:
        return
    LOCAL_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        series = cam["series"]
        if int(np.isfinite(series["ci"]).sum()) < 20:
            continue
        record_key = key if len(cameras) == 1 else f"{key}-cam{cam['id']}"
        record = series_record(series, video_key=record_key, filename=filename)
        record["source_video_keys"] = [key]
        target = LOCAL_LIBRARY_DIR / f"{record_key}.json"
        target.with_suffix(".tmp").write_text(json.dumps(record), encoding="utf-8")
        target.with_suffix(".tmp").replace(target)
    for old in sorted(LOCAL_LIBRARY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[LOCAL_LIBRARY_MAX:]:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------- jobs

def _now() -> float:
    return time.time()


def _busy() -> bool:
    return any(job["status"] in {"queued", "running"} for job in JOBS.values())


def _prune_jobs() -> None:
    for job_id, job in list(JOBS.items()):
        if job["status"] in {"succeeded", "failed"} and _now() - job.get("finished", _now()) > JOB_RETENTION_SECONDS:
            JOBS.pop(job_id, None)
            shutil.rmtree(EVIDENCE_DIR / job_id, ignore_errors=True)


def _run_job(job_id: str, path: Path, filename: str, timing_confirmed: bool) -> None:
    def progress(pct: int, done: int, total: int) -> None:
        with LOCK:
            JOBS[job_id].update(progress_percent=pct, frames_decoded=done, frames_total=total)
    try:
        with LOCK:
            JOBS[job_id]["status"] = "running"
        detector.load()
        key = video_key(path)
        library, library_videos = load_library(key)
        analysis = analyze_video(path, detector, max_duration_seconds=MAX_VIDEO_SECONDS,
                                 timing_confirmed=timing_confirmed, sample_fps=SAMPLE_FPS,
                                 library=library, library_videos=library_videos, progress=progress)
        try:
            save_to_library(key, filename, analysis["cameras"])
        except OSError:
            logger.warning("Could not save the CI series to the local library", exc_info=True)
        result = result_payload(analysis, filename)
        try:
            frames = capture_evidence(path, detector, analysis["cameras"], EVIDENCE_DIR / job_id)
            result["evidence"] = [{**{k: v for k, v in f.items() if k != "file"},
                                   "url": f"/api/jobs/{job_id}/evidence/{f['file']}"} for f in frames]
        except Exception:
            logger.warning("Could not capture evidence frames", exc_info=True)
            result["evidence"] = []
        with LOCK:
            JOBS[job_id].update(status="succeeded", progress_percent=100, result=result)
    except VideoAnalysisError as exc:
        with LOCK:
            JOBS[job_id].update(status="failed", error=str(exc))
    except Exception:
        logger.exception("Analysis job %s failed", job_id)
        with LOCK:
            JOBS[job_id].update(status="failed", error="Không phân tích được video. Vui lòng thử lại hoặc chọn video khác.")
    finally:
        path.unlink(missing_ok=True)
        with LOCK:
            JOBS[job_id]["finished"] = _now()


def _check_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if not ok or fps <= 0 or frames <= 0:
        raise HTTPException(400, "Không đọc được video hoặc FPS không hợp lệ.")
    if frames / fps > MAX_VIDEO_SECONDS:
        raise HTTPException(413, f"Video vượt giới hạn {MAX_VIDEO_SECONDS // 60} phút.")


# ---------------------------------------------------------------- routes

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/map", include_in_schema=False)
def map_page() -> RedirectResponse:
    return RedirectResponse("/#overview")


@app.get("/api/map/config")
def map_config() -> dict[str, Any]:
    tiles = os.getenv("MAP_TILE_URL_TEMPLATE", "").strip()
    if not (tiles.startswith("https://") and all(p in tiles for p in ("{z}", "{x}", "{y}"))):
        tiles = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
    else:
        attribution = os.getenv("MAP_ATTRIBUTION", "").strip()[:300]
    return {"tile_url_template": tiles, "attribution": attribution, "center": [21.0285, 105.8542],
            "boundary_url": "/static/hanoi_boundary.geojson",
            "areas": [{"name": n, "latitude": lat, "longitude": lon} for n, lat, lon in weather.AREAS],
            # Used by the page when the main tile server refuses requests (OSM blocks some clients).
            "fallback_tile_url_template": "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            "fallback_attribution": '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> · nền HOT, OpenStreetMap France'}


@app.get("/api/weather")
def weather_point(latitude: float = 21.0285, longitude: float = 105.8542) -> dict[str, Any]:
    """Now, next 24 hours and 3 days at one point in Hanoi."""
    try:
        return weather.forecast(latitude, longitude)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except weather.WeatherError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/weather/overview")
def weather_overview() -> dict[str, Any]:
    """Current weather at one point per area across Hanoi."""
    try:
        return weather.overview()
    except weather.WeatherError as exc:
        raise HTTPException(502, str(exc)) from exc


def _point(text: str, name: str) -> tuple[float, float]:
    try:
        lat, lon = (float(v) for v in text.split(","))
    except ValueError as exc:
        raise HTTPException(422, f"{name} phải có dạng 'vĩ độ,kinh độ'.") from exc
    if not is_hanoi_coordinate(lat, lon):
        raise HTTPException(422, f"{name} nằm ngoài ranh giới Hà Nội.")
    return lat, lon


@app.get("/api/routes")
def routes(from_: str = Query(..., alias="from"), to: str = Query(...)) -> dict[str, Any]:
    """Up to three driving routes between two points in Hanoi, fastest first (OSRM, no live traffic)."""
    origin, destination = _point(from_, "Điểm đi"), _point(to, "Điểm đến")
    try:
        found = routing.route_alternatives(origin, destination)
    except routing.RouteError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"routes": found, "source": "OSRM · OpenStreetMap",
            "note": "Thời gian theo mạng đường OSRM/OpenStreetMap, chưa tính tắc đường thời gian thực."}


@app.get("/api/model/report")
def model_report() -> dict[str, Any]:
    """Leave-one-video-out evaluation of the shipped training library (tools/build_forecast_library.py)."""
    path = ROOT / "artifacts" / "forecast_library_report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Chưa có báo cáo đánh giá mô hình.") from exc


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    """Saved analysis of a held-out Hanoi recording, played from its source segments."""
    try:
        config = json.loads((ROOT / "artifacts" / "demo.json").read_text(encoding="utf-8"))
        session = json.loads((ROOT / "artifacts" / config["session_file"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(404, "Chưa cấu hình demo.") from exc
    start, segments = 0.0, []
    for seg in config["segments"]:
        segments.append({"name": seg["name"], "start": start, "duration": seg["duration"],
                         "download_url": f"https://drive.google.com/file/d/{seg['drive_id']}/view"})
        start += float(seg["duration"])
    return {"title": config["title"], "note": config["note"], "source": config.get("source"),
            "segments": segments, "session": session}


LIVE = LiveStore(lambda: load_library(None), location_ok=is_hanoi_coordinate)


@app.post("/api/live/{camera_id}/samples")
def live_samples(camera_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Collector intake: 1 Hz CI measurements of one camera (authenticated by X-Live-Token)."""
    try:
        return LIVE.observe(camera_id, body.get("samples") or [], name=body.get("name"),
                            lat=body.get("lat"), lng=body.get("lng"))
    except LiveError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/live/{camera_id}")
def live_remove(camera_id: str) -> dict[str, Any]:
    if not LIVE.remove(camera_id):
        raise HTTPException(404, "Không có camera này.")
    return {"removed": camera_id}


@app.get("/api/live")
def live_list() -> dict[str, Any]:
    cameras = [LIVE.forecast(cid) for cid in LIVE.camera_ids()]
    return {"enabled": bool(LIVE_TOKEN), "cameras": [c for c in cameras if c]}


@app.get("/api/live/{camera_id}")
def live_camera(camera_id: str) -> dict[str, Any]:
    found = LIVE.forecast(camera_id)
    if found is None:
        raise HTTPException(404, "Chưa có số liệu từ camera này.")
    return found


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    with LOCK:
        busy = _busy()
    _, library_videos = load_library(None)
    ready = detector.available
    return {
        "ready": ready,
        "busy": busy,
        "message": None if ready else "Thiếu models/detector.onnx.",
        "upload_limit_mib": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_duration_seconds": MAX_VIDEO_SECONDS,
        "library_videos": library_videos,
        "token_required": bool(UPLOAD_TOKEN),
        "live_enabled": bool(LIVE_TOKEN),
    }


@app.post("/api/analyze", status_code=202)
async def analyze(file: UploadFile = File(...), timing_confirmed: bool = Form(False)) -> dict[str, Any]:
    # Token, hourly limit, busy state and size are checked by UploadGuard before this body is parsed.
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, "Định dạng hỗ trợ: " + ", ".join(sorted(s[1:] for s in SUPPORTED_SUFFIXES)))
    if not detector.available:
        raise HTTPException(503, "Thiếu models/detector.onnx.")
    with LOCK:
        if _busy():
            raise HTTPException(409, "Máy chủ đang phân tích một video khác. Vui lòng thử lại sau.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(UPLOAD_DIR).free < (file.size or MAX_UPLOAD_BYTES) + DISK_RESERVE_BYTES:
        raise HTTPException(507, "Không đủ dung lượng đĩa để nhận video.")
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, prefix="upload_", suffix=suffix, delete=False) as tmp:
            path = Path(tmp.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Video vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.")
                await run_in_threadpool(tmp.write, chunk)
        if size == 0:
            raise HTTPException(400, "Tệp video rỗng.")
        await run_in_threadpool(_check_video, path)
        with LOCK:
            if _busy():
                raise HTTPException(409, "Máy chủ đang phân tích một video khác. Vui lòng thử lại sau.")
            _prune_jobs()
            job_id = uuid.uuid4().hex
            JOBS[job_id] = {"job_id": job_id, "status": "queued", "filename": filename,
                            "progress_percent": 0, "frames_decoded": 0, "frames_total": 0}
            EXECUTOR.submit(_run_job, job_id, path, filename, timing_confirmed)
        path = None
        return {"job_id": job_id, "status": "queued"}
    finally:
        await file.close()
        if path is not None:
            path.unlink(missing_ok=True)


@app.get("/api/jobs/{job_id}/evidence/{name}")
def evidence_frame(job_id: str, name: str) -> FileResponse:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not re.fullmatch(r"\d{2}\.jpg", name):
        raise HTTPException(404, "Không tìm thấy ảnh.")
    path = EVIDENCE_DIR / job_id / name
    if not path.is_file():
        raise HTTPException(404, "Ảnh không còn (tác vụ đã hết hạn).")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Không tìm thấy tác vụ.")
    with LOCK:
        _prune_jobs()
        found = JOBS.get(job_id)
        if found is None:
            raise HTTPException(404, "Không tìm thấy tác vụ hoặc tác vụ đã hết hạn.")
        return {k: v for k, v in found.items() if k != "finished"}
