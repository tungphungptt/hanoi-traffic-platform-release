from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager
import math
import mimetypes
import os
import random
import re
import shutil
import tempfile
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from app.public_upload import PublicUploadMiddleware, UploadGate, BUSY_MESSAGE
import cv2
from pydantic import BaseModel, Field, field_validator

from aicc.pipeline import MAX_VIDEO_BYTES, SUPPORTED_VIDEO_EXTENSIONS, aicc_runtime, process_video
from traffic_brain.providers import ForecastProviderManager, load_dotenv
from traffic_brain.feature_engine import FeatureContractError, aggregate_traffic_features
from traffic_brain.analytics import AnalyticsInputError, TrafficAnalyticsEngine
from traffic_brain.video_review import analyze_video, VideoReviewError
from traffic_brain.hanoi_video_forecaster import hanoi_video_forecaster
from traffic_brain.routing import RouteServiceError, fetch_route_alternatives, score_routes
from app.weather_service import WeatherServiceError, get_weather_report
from app.traffic_baseline import forecast_from_history
from app.hanoi_scope import is_hanoi_coordinate

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
ARTIFACT_DIR = ROOT / "artifacts" / "current"
DATA_DIR = ROOT / "data"
CAMERA_CONFIG_PATH = DATA_DIR / "cameras.json"
CAMERA_FEED_DIR = DATA_DIR / "camera_feeds"
CAMERA_FEED_INDEX_PATH = CAMERA_FEED_DIR / "index.json"
MAX_CAMERA_FEED_BYTES = 500 * 1024 * 1024
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_FEED_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
logger = logging.getLogger("hanoi_traffic")
PUBLIC_DEMO_MODE = os.getenv("PUBLIC_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
AICC_VIDEO_ENABLED = os.getenv("AICC_VIDEO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
PUBLIC_VIDEO_REVIEW_ENABLED = os.getenv("PUBLIC_VIDEO_REVIEW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
VIDEO_REVIEW_ROOT = DATA_DIR / "video_reviews"
VIDEO_REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_REVIEW_UPLOAD_ROOT = DATA_DIR / "uploads" / "video_reviews"
VIDEO_REVIEW_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE_REVIEW_UPLOADS: set[Path] = set()
MAX_VIDEO_REVIEW_BYTES = 500 * 1024 * 1024
MAX_PUBLIC_VIDEO_REVIEW_BYTES = 100 * 1024 * 1024
MAX_PUBLIC_VIDEO_REVIEW_DURATION_SECONDS = 5 * 60
MAX_VIDEO_REVIEW_DURATION_SECONDS = 20 * 60


def video_review_upload_enabled() -> bool:
    """Allow only video-review uploads in public mode when explicitly enabled."""
    if PUBLIC_DEMO_MODE:
        return PUBLIC_VIDEO_REVIEW_ENABLED
    return AICC_VIDEO_ENABLED


def require_write_access() -> None:
    """In public demo mode, keep the site read-only for anonymous visitors."""
    if PUBLIC_DEMO_MODE:
        raise HTTPException(status_code=403, detail="Public demo đang ở chế độ chỉ đọc; thao tác ghi đã bị tắt.")


def video_review_access() -> dict[str, Any]:
    """Expose the effective server flags, independently of model forecast readiness."""
    return {
        "public_demo_mode": PUBLIC_DEMO_MODE,
        "video_enabled": AICC_VIDEO_ENABLED,
        "enabled": AICC_VIDEO_ENABLED and not PUBLIC_DEMO_MODE,
        "blocked_reason": "public_demo" if PUBLIC_DEMO_MODE else "video_disabled" if not AICC_VIDEO_ENABLED else None,
    }

DEFAULT_MANIFEST: dict[str, Any] = {
    "artifact_version": "2.0.0-schema-only",
    "domain_profile": "SEA_Urban_Unstructured_Hanoi",
    "detector_config": {
        "onnx_path": "artifacts/current/detector.onnx",
        "input_size": [640, 640],
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "class_mapping": {
            "0": {"name": "motorcycle", "pcu_weight": 0.3, "avg_area_m2": 1.6},
            "1": {"name": "car", "pcu_weight": 1.0, "avg_area_m2": 8.5},
            "2": {"name": "bus", "pcu_weight": 2.5, "avg_area_m2": 24.0},
            "3": {"name": "truck", "pcu_weight": 2.0, "avg_area_m2": 18.0},
        },
    },
    "camera_calibration": {
        "effective_road_length_m": 45.0,
        "effective_road_area_m2": 315.0,
        "free_flow_speed_kmh": 42.0,
    },
    "feature_schema": {
        "window_seconds": 60,
        "slide_step_seconds": 15,
        "ordered_input_keys": [
            "flow.car_count",
            "flow.motorcycle_count",
            "flow.pcu_equivalent",
            "speed.mean_kmh",
            "speed.cv_ratio",
            "density.areal_occupancy",
            "temporal.pcu_lag_1",
            "temporal.speed_delta_1",
        ],
        "default_fallback_values": {
            "flow.car_count": 0.0,
            "flow.motorcycle_count": 0.0,
            "flow.pcu_equivalent": 0.0,
            "speed.mean_kmh": 42.0,
            "speed.cv_ratio": 0.0,
            "density.areal_occupancy": 0.0,
            "temporal.pcu_lag_1": 0.0,
            "temporal.speed_delta_1": 0.0,
        },
    },
    "forecaster_config": {
        "onnx_path": "artifacts/current/forecaster.onnx",
        "horizons_minutes": [5, 10, 15],
        "thresholds": {"busy": 0.30, "heavy": 0.55, "gridlock": 0.78},
    },
}


def _finite_number(value: Any, field_name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} phải là số, không phải boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} phải là số hữu hạn") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} phải là số hữu hạn")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} phải >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} phải <= {maximum}")
    return number


def _normalize_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the versioned model contract and constrain artifact paths to this release tree."""
    if not isinstance(payload, dict):
        raise ValueError("manifest phải là một JSON object")
    # JSON round-trip makes a detached, JSON-safe copy; callers' dicts are never mutated.
    try:
        result = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest chỉ được chứa kiểu dữ liệu JSON hợp lệ và số hữu hạn") from exc

    result.setdefault("artifact_version", "2.0.0-schema-only")
    result.setdefault("domain_profile", "SEA_Urban_Unstructured_Hanoi")
    for field_name in ("artifact_version", "domain_profile"):
        field_value = result.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip() or len(field_value) > 120:
            raise ValueError(f"{field_name} phải là chuỗi không rỗng (tối đa 120 ký tự)")
        result[field_name] = field_value.strip()

    detector = result.setdefault("detector_config", {})
    forecaster = result.setdefault("forecaster_config", {})
    schema = result.get("feature_schema")
    if not isinstance(detector, dict) or not isinstance(forecaster, dict) or not isinstance(schema, dict):
        raise ValueError("manifest cần detector_config, forecaster_config và feature_schema dạng object")

    # Paths may be a basename or the project's canonical relative path only. Absolute paths,
    # drive letters and parent traversal are rejected, rather than silently normalized.
    for section, expected_name in (("detector_config", "detector.onnx"), ("forecaster_config", "forecaster.onnx")):
        raw_path = result[section].get("onnx_path", f"artifacts/current/{expected_name}")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{section}.onnx_path phải là chuỗi")
        candidate = raw_path.strip().replace("\\", "/")
        parts = candidate.split("/")
        if candidate.startswith("/") or ":" in candidate or ".." in parts or any(part in {"", "."} for part in parts):
            raise ValueError(f"{section}.onnx_path không được là đường dẫn tuyệt đối hoặc chứa traversal")
        if candidate not in {expected_name, f"artifacts/current/{expected_name}"}:
            raise ValueError(f"{section}.onnx_path chỉ được trỏ tới {expected_name} trong artifacts/current/")
        result[section]["onnx_path"] = f"artifacts/current/{expected_name}"

    input_size = detector.setdefault("input_size", [640, 640])
    if (not isinstance(input_size, list) or len(input_size) != 2 or
            any(not isinstance(n, int) or isinstance(n, bool) or n < 32 or n > 2048 for n in input_size)):
        raise ValueError("detector_config.input_size phải gồm 2 số nguyên trong khoảng 32–2048")
    for key in ("conf_threshold", "iou_threshold"):
        value = _finite_number(detector.setdefault(key, 0.35 if key == "conf_threshold" else 0.45), f"detector_config.{key}", minimum=0, maximum=1)
        detector[key] = value

    ordered_keys = schema.get("ordered_input_keys")
    if not isinstance(ordered_keys, list) or not 1 <= len(ordered_keys) <= 128:
        raise ValueError("feature_schema.ordered_input_keys phải là danh sách 1–128 đặc trưng")
    if any(not isinstance(key, str) or not key.strip() or len(key) > 160 for key in ordered_keys):
        raise ValueError("Mỗi ordered_input_key phải là chuỗi không rỗng, tối đa 160 ký tự")
    cleaned_keys = [key.strip() for key in ordered_keys]
    if len(set(cleaned_keys)) != len(cleaned_keys):
        raise ValueError("feature_schema.ordered_input_keys không được lặp khóa")
    schema["ordered_input_keys"] = cleaned_keys
    fallback = schema.setdefault("default_fallback_values", {})
    if not isinstance(fallback, dict):
        raise ValueError("feature_schema.default_fallback_values phải là object")
    normalized_fallback: dict[str, float] = {}
    for key, value in fallback.items():
        if key not in schema["ordered_input_keys"]:
            raise ValueError(f"Giá trị fallback '{key}' không có trong ordered_input_keys")
        normalized_fallback[key] = _finite_number(value, f"default_fallback_values.{key}")
    schema["default_fallback_values"] = normalized_fallback
    schema["window_seconds"] = int(_finite_number(schema.setdefault("window_seconds", 60), "window_seconds", minimum=1, maximum=3600))
    schema["slide_step_seconds"] = int(_finite_number(schema.setdefault("slide_step_seconds", 15), "slide_step_seconds", minimum=1, maximum=3600))

    horizons = forecaster.setdefault("horizons_minutes", [5, 10, 15])
    if horizons != [5, 10, 15]:
        raise ValueError("forecaster_config.horizons_minutes phải đúng thứ tự [5, 10, 15]")
    thresholds = forecaster.setdefault("thresholds", {"busy": 0.30, "heavy": 0.55, "gridlock": 0.78})
    if not isinstance(thresholds, dict):
        raise ValueError("forecaster_config.thresholds phải là object")
    cleaned_thresholds = {name: _finite_number(thresholds.get(name, default), f"thresholds.{name}", minimum=0, maximum=1)
                          for name, default in (("busy", 0.30), ("heavy", 0.55), ("gridlock", 0.78))}
    if not cleaned_thresholds["busy"] < cleaned_thresholds["heavy"] < cleaned_thresholds["gridlock"]:
        raise ValueError("Ngưỡng CI phải tăng theo thứ tự busy < heavy < gridlock")
    forecaster["thresholds"] = cleaned_thresholds

    calibration = result.setdefault("camera_calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError("camera_calibration phải là object")
    for key, default in (("effective_road_length_m", 45.0), ("effective_road_area_m2", 315.0), ("free_flow_speed_kmh", 42.0)):
        calibration[key] = _finite_number(calibration.get(key, default), f"camera_calibration.{key}", minimum=0.01, maximum=100000)
    class_mapping = detector.setdefault("class_mapping", {})
    if not isinstance(class_mapping, dict):
        raise ValueError("detector_config.class_mapping phải là object")
    return result


def _load_manifest_file() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(DEFAULT_MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
        return _normalize_manifest_payload(payload)
    except Exception as exc:
        raise RuntimeError(f"Không đọc được manifest.json: {exc}") from exc


def _artifact_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        logger.warning("Bỏ qua đường dẫn artifact ngoài thư mục dự án")
        return None
    return resolved


class ArtifactRuntime:
    """Đọc manifest động và nạp mô hình ONNX trên CPU. Thiếu artifact thì chạy demo rõ ràng."""

    def __init__(self) -> None:
        self.manifest: dict[str, Any] = {}
        self.detector_session = None
        self.forecaster_session = None
        self.detector_error: str | None = None
        self.forecaster_error: str | None = None
        self.loaded_at: str | None = None
        self.runtime_provider = "CPUExecutionProvider"
        self.reload()

    def reload(self) -> dict[str, Any]:
        self.manifest = _load_manifest_file()
        self.detector_session = None
        self.forecaster_session = None
        self.detector_error = None
        self.forecaster_error = None
        try:
            import onnxruntime as ort  # type: ignore
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            detector_path = _artifact_path(self.manifest.get("detector_config", {}).get("onnx_path"))
            forecaster_path = _artifact_path(self.manifest.get("forecaster_config", {}).get("onnx_path"))
            # Load each artifact independently: a malformed detector must not block a valid forecaster.
            if detector_path and detector_path.exists():
                try:
                    self.detector_session = ort.InferenceSession(str(detector_path), sess_options=options, providers=["CPUExecutionProvider"])
                except Exception as exc:
                    self.detector_error = str(exc)[:240]
            if forecaster_path and forecaster_path.exists():
                try:
                    self.forecaster_session = ort.InferenceSession(str(forecaster_path), sess_options=options, providers=["CPUExecutionProvider"])
                except Exception as exc:
                    self.forecaster_error = str(exc)[:240]
        except ImportError:
            self.detector_error = "Chưa cài onnxruntime; cài requirements.txt để nạp mô hình ONNX."
            self.forecaster_error = self.detector_error
        except Exception as exc:
            # Keep the server available so the operator can repair/replace an invalid artifact.
            message = str(exc)[:240]
            self.detector_error = self.detector_error or message
            self.forecaster_error = self.forecaster_error or message
        self.loaded_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return self.status()

    def status(self) -> dict[str, Any]:
        det_cfg = self.manifest.get("detector_config", {})
        fore_cfg = self.manifest.get("forecaster_config", {})
        det_path = _artifact_path(det_cfg.get("onnx_path"))
        fore_path = _artifact_path(fore_cfg.get("onnx_path"))
        return {
            "artifact_version": self.manifest.get("artifact_version", "unknown"),
            "domain_profile": self.manifest.get("domain_profile", "unknown"),
            "mode": "production" if self.forecaster_session else "demo",
            "detector": {
                "ready": self.detector_session is not None,
                "file": det_path.name if det_path else None,
                "exists": bool(det_path and det_path.exists()),
                "error": self.detector_error,
            },
            "forecaster": {
                "ready": self.forecaster_session is not None,
                "file": fore_path.name if fore_path else None,
                "exists": bool(fore_path and fore_path.exists()),
                "error": self.forecaster_error,
            },
            "provider": self.runtime_provider,
            "loaded_at": self.loaded_at,
            "feature_count": len(self.manifest.get("feature_schema", {}).get("ordered_input_keys", [])),
            "horizons_minutes": self.manifest.get("forecaster_config", {}).get("horizons_minutes", [5, 10, 15]),
        }

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        schema = self.manifest.get("feature_schema", {})
        ordered = schema.get("ordered_input_keys", [])
        fallback = schema.get("default_fallback_values", {})
        values = [float(features.get(key, fallback.get(key, 0.0))) for key in ordered]
        horizons = self.manifest.get("forecaster_config", {}).get("horizons_minutes", [5, 10, 15])
        thresholds = self.manifest.get("forecaster_config", {}).get("thresholds", {"busy": 0.30, "heavy": 0.55, "gridlock": 0.78})

        if self.forecaster_session is not None:
            try:
                import numpy as np  # type: ignore
                session = self.forecaster_session
                in_meta = session.get_inputs()[0]
                model_input = np.asarray([values], dtype=np.float32)
                # Most tabular exports accept [1, n_features]. Reshape supports a fixed flat input if exported that way.
                expected = list(in_meta.shape)
                if len(expected) == 1:
                    model_input = np.asarray(values, dtype=np.float32).reshape(-1)
                raw = session.run(None, {in_meta.name: model_input})[0]
                flat_array = np.asarray(raw, dtype=float).reshape(-1)
                if flat_array.size < len(horizons):
                    raise ValueError(
                        f"Forecaster trả {flat_array.size} giá trị nhưng manifest yêu cầu "
                        f"{len(horizons)} chân trời dự báo; từ chối lặp một giá trị cho nhiều mốc thời gian."
                    )
                if not np.all(np.isfinite(flat_array)):
                    raise ValueError("Forecaster trả NaN/Inf; kiểm tra mô hình và đầu vào")
                flat = flat_array.tolist()
                preds = [max(0.0, min(1.0, float(flat[i]))) for i in range(len(horizons))]
                mode = "production"
            except Exception as exc:
                return {"mode": "error", "error": f"Không suy luận được forecaster.onnx: {str(exc)[:220]}", "predictions": []}
        else:
            # Demo formula only: intentionally labelled as illustrative, never represented as a trained-model output.
            speed = max(0.0, min(60.0, float(features.get("speed.mean_kmh", 25.0))))
            occ = max(0.0, min(1.0, float(features.get("density.areal_occupancy", 0.34))))
            cv = max(0.0, min(3.0, float(features.get("speed.cv_ratio", 0.55))))
            pcu = max(0.0, float(features.get("flow.pcu_equivalent", 60.0)))
            base = 0.50 * (1 - min(speed / 42.0, 1.0)) + 0.35 * min(occ / 0.65, 1.0) + 0.15 * min(cv / 1.8, 1.0)
            base += min(pcu / 280.0, 0.12) * 0.20
            base = max(0.04, min(0.96, base))
            # These trajectories are illustrative only; real mode comes exclusively from ONNX outputs.
            preds = [max(0.0, min(1.0, base + offset)) for offset in (0.0, 0.035, 0.07)]
            mode = "demo"

        def label(ci: float) -> tuple[str, str]:
            if ci < thresholds.get("busy", 0.30):
                return "Thông thoáng", "free"
            if ci < thresholds.get("heavy", 0.55):
                return "Đông đúc", "busy"
            if ci < thresholds.get("gridlock", 0.78):
                return "Ùn ứ cục bộ", "heavy"
            return "Tắc nghẽn nghiêm trọng", "gridlock"

        items = []
        for idx, horizon in enumerate(horizons):
            ci = round(float(preds[idx]), 3)
            name, key = label(ci)
            items.append({"horizon_minutes": int(horizon), "ci": ci, "level": name, "level_key": key})
        return {"mode": mode, "predictions": items, "features_used": dict(zip(ordered, values))}


runtime = ArtifactRuntime()
forecast_providers = ForecastProviderManager(runtime, MANIFEST_PATH)


@asynccontextmanager
async def app_lifespan(app):
    async def maintain_reviews():
        while True:
            try:
                await run_in_threadpool(_cleanup_video_reviews)
            except Exception:
                logger.exception("Video review retention cleanup failed")
            await asyncio.sleep(60)
    task = asyncio.create_task(maintain_reviews())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


PUBLIC_UPLOAD_GATE = UploadGate()
app = FastAPI(lifespan=app_lifespan, title="Hanoi Traffic Operations", version="8.0.0", description="Nền tảng giám sát và hỗ trợ dự báo giao thông trên CPU")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.add_middleware(PublicUploadMiddleware, public_mode=lambda: PUBLIC_DEMO_MODE,
                   enabled=video_review_upload_enabled, max_bytes=lambda: MAX_PUBLIC_VIDEO_REVIEW_BYTES,
                   busy=lambda: _active_aicc_job_exists(), gate=lambda: PUBLIC_UPLOAD_GATE)



@app.middleware("http")
async def video_review_cache_policy(request: Request, call_next):
    response = await call_next(request)
    if (request.url.path in {"/static/app.js", "/api/health", "/api/aicc/status", "/api/aicc/data-videos"}
            or request.url.path.startswith("/api/scenarios/video-reviews/")):
        response.headers["Cache-Control"] = "no-store"
    return response


DEFAULT_CAMERAS: list[dict[str, Any]] = [
    {"id":"HN-001","name":"Ngã Tư Sở","area":"Đống Đa · Thanh Xuân","lat":21.002,"lng":105.820,"map_x":57,"map_y":54,"status":"online","vehicles":128,"speed":18.6,"occupancy":0.48,"ci":0.48,"updated":"Dữ liệu mẫu"},
    {"id":"HN-002","name":"Cầu Giấy – Xuân Thủy","area":"Cầu Giấy","lat":21.036,"lng":105.790,"map_x":42,"map_y":25,"status":"online","vehicles":156,"speed":12.4,"occupancy":0.67,"ci":0.69,"updated":"Dữ liệu mẫu"},
    {"id":"HN-003","name":"Nguyễn Trãi – Khuất Duy Tiến","area":"Thanh Xuân","lat":20.998,"lng":105.803,"map_x":36,"map_y":58,"status":"online","vehicles":184,"speed":8.2,"occupancy":0.73,"ci":0.82,"updated":"Dữ liệu mẫu"},
    {"id":"HN-004","name":"Trần Duy Hưng","area":"Cầu Giấy","lat":21.008,"lng":105.800,"map_x":63,"map_y":31,"status":"online","vehicles":94,"speed":24.1,"occupancy":0.39,"ci":0.34,"updated":"Dữ liệu mẫu"},
    {"id":"HN-005","name":"Vành đai 3 – Pháp Vân","area":"Hoàng Mai","lat":20.973,"lng":105.840,"map_x":73,"map_y":77,"status":"attention","vehicles":202,"speed":7.1,"occupancy":0.81,"ci":0.87,"updated":"Dữ liệu mẫu"},
    {"id":"HN-006","name":"Kim Mã – Giảng Võ","area":"Ba Đình","lat":21.030,"lng":105.820,"map_x":23,"map_y":31,"status":"online","vehicles":82,"speed":29.5,"occupancy":0.26,"ci":0.22,"updated":"Dữ liệu mẫu"},
    {"id":"HN-007","name":"Láng – Yên Lãng","area":"Đống Đa","lat":21.014,"lng":105.810,"map_x":28,"map_y":45,"status":"offline","vehicles":0,"speed":0,"occupancy":0,"ci":0,"updated":"Dữ liệu mẫu"},
    {"id":"HN-008","name":"Long Biên – Nguyễn Văn Cừ","area":"Long Biên","lat":21.047,"lng":105.875,"map_x":83,"map_y":44,"status":"online","vehicles":106,"speed":21.6,"occupancy":0.35,"ci":0.31,"updated":"Dữ liệu mẫu"},
]


def _load_cameras() -> tuple[list[dict[str, Any]], str | None]:
    """Load operator-editable demo camera metadata without trusting arbitrary paths."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CAMERA_CONFIG_PATH.exists():
        CAMERA_CONFIG_PATH.write_text(json.dumps({"description":"Dữ liệu camera minh họa; hãy thay bằng metadata camera được cấp phép.","cameras":DEFAULT_CAMERAS}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        payload = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
        entries = payload.get("cameras") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("data/cameras.json cần có mảng cameras không rỗng")
        known = {str(row["id"]): row for row in DEFAULT_CAMERAS}
        loaded: list[dict[str, Any]] = []
        seen: set[str] = set()
        allowed_status = {"online", "attention", "offline"}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Mỗi camera phải là JSON object")
            cid = str(entry.get("id", "")).strip().upper()
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{1,31}", cid):
                raise ValueError(f"Mã camera không hợp lệ: {cid!r}")
            if cid in seen:
                raise ValueError(f"Mã camera bị trùng: {cid}")
            seen.add(cid)
            base = dict(known.get(cid, {"id":cid,"name":cid,"area":"Chưa cấu hình khu vực","lat":None,"lng":None,"map_x":50,"map_y":50,"status":"offline","vehicles":0,"speed":0,"occupancy":0,"ci":0,"updated":"Chưa có dữ liệu"}))
            base.update(entry)
            base["id"] = cid
            base["name"] = str(base.get("name") or cid)[:120]
            base["area"] = str(base.get("area") or "Chưa cấu hình khu vực")[:120]
            if base.get("status") not in allowed_status:
                raise ValueError(f"Trạng thái camera {cid} phải là online/attention/offline")
            for key, lo, hi in (("map_x", 3, 97), ("map_y", 3, 97), ("ci", 0, 1), ("occupancy", 0, 1), ("speed", 0, 250), ("vehicles", 0, 100000)):
                try: val = float(base.get(key, lo))
                except (TypeError, ValueError) as exc: raise ValueError(f"Giá trị {key} không hợp lệ ở {cid}") from exc
                if not math.isfinite(val) or val < lo or val > hi:
                    raise ValueError(f"Giá trị {key} ngoài giới hạn ở {cid}")
                base[key] = int(val) if key == "vehicles" else val
            loaded.append(base)
        return loaded, None
    except Exception as exc:
        logger.exception("Không đọc được data/cameras.json; dùng cấu hình fixture tích hợp")
        return [dict(row) for row in DEFAULT_CAMERAS], f"Không đọc được data/cameras.json: {str(exc)[:180]}"


CAMERAS, CAMERA_CONFIG_WARNING = _load_cameras()


ALERTS = [
    {"id": "AL-2601", "type": "critical", "title": "Nguy cơ tắc nghẽn nghiêm trọng", "location": "Nguyễn Trãi – Khuất Duy Tiến", "camera_id": "HN-003", "time": "2 phút trước", "ci": 0.82, "message": "Tốc độ giảm, độ chiếm dụng mặt đường tăng. Dự báo mức ùn tắc cao trong 5–10 phút tới.", "status": "Chưa xử lý"},
    {"id": "AL-2602", "type": "warning", "title": "Lưu lượng tăng nhanh", "location": "Cầu Giấy – Xuân Thủy", "camera_id": "HN-002", "time": "6 phút trước", "ci": 0.69, "message": "Dòng xe có dấu hiệu Stop-and-Go. Nên theo dõi diễn biến ở mốc 10–15 phút.", "status": "Đang theo dõi"},
    {"id": "AL-2603", "type": "warning", "title": "Mật độ cao tại cửa ngõ", "location": "Vành đai 3 – Pháp Vân", "camera_id": "HN-005", "time": "9 phút trước", "ci": 0.87, "message": "Camera ghi nhận mật độ phương tiện cao, tốc độ trung bình thấp.", "status": "Đang theo dõi"},
    {"id": "AL-2604", "type": "info", "title": "Lưu thông ổn định trở lại", "location": "Trần Duy Hưng", "camera_id": "HN-004", "time": "18 phút trước", "ci": 0.34, "message": "Chỉ số ùn tắc giảm so với cửa sổ trước.", "status": "Đã ghi nhận"},
]


def _read_camera_feed_index() -> dict[str, str]:
    try:
        payload = json.loads(CAMERA_FEED_INDEX_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict): return {}
        clean: dict[str, str] = {}
        allowed_ids = {str(camera["id"]) for camera in CAMERAS}
        for cid, filename in payload.items():
            if cid in allowed_ids and isinstance(filename, str) and Path(filename).name == filename and Path(filename).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                clean[cid] = filename
        return clean
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Không đọc được index video camera: %s", exc)
        return {}


def _camera_feed_path(camera_id: str) -> tuple[Path | None, str]:
    """Return only a clip explicitly assigned to this camera ID; never mislabel one shared clip as every road."""
    CAMERA_FEED_DIR.mkdir(parents=True, exist_ok=True)
    index = _read_camera_feed_index()
    assigned_name = index.get(camera_id)
    if assigned_name:
        candidate = (CAMERA_FEED_DIR / assigned_name).resolve()
        try:
            candidate.relative_to(CAMERA_FEED_DIR.resolve())
            if candidate.is_file(): return candidate, "assigned_clip"
        except ValueError:
            pass
    return None, "none"


def _camera_feed_info(camera_id: str) -> dict[str, Any]:
    path, source_type = _camera_feed_path(camera_id)
    if path is None:
        return {"available": False, "source_type": "none", "filename": None, "size_mb": 0, "stream_url": None,
                "notice": "Chưa gán clip cho điểm này. Ảnh minh họa không phải CCTV trực tiếp; hãy gán video riêng cho đúng camera."}
    size = path.stat().st_size
    version = path.stat().st_mtime_ns
    url = f"/api/cameras/{camera_id}/feed?v={version}"
    notice = "Clip đã gán cho đúng mã camera trong bản demo; chưa phải luồng CCTV/RTSP trực tiếp."
    return {"available": True, "source_type": source_type, "filename": path.name, "size_mb": round(size/(1024*1024), 2), "stream_url": url, "notice": notice}


def _assert_camera_exists(camera_id: str) -> None:
    if not any(str(camera["id"]) == camera_id for camera in CAMERAS):
        raise HTTPException(status_code=404, detail="Không tìm thấy điểm camera. Kiểm tra data/cameras.json rồi khởi động lại server.")


def _save_camera_feed_index(index: dict[str, str]) -> None:
    CAMERA_FEED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CAMERA_FEED_INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CAMERA_FEED_INDEX_PATH)


def _video_range_response(request: Request, path: Path) -> StreamingResponse:
    """Serve a local clip with HTTP byte ranges so the browser can seek without downloading it all."""
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    start, end, status_code = 0, max(0, size - 1), 200
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store", "Content-Disposition": f'inline; filename="{path.name}"'}
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or size == 0:
            return StreamingResponse(iter(()), status_code=416, media_type=media_type, headers={**headers, "Content-Range": f"bytes */{size}"})
        left, right = match.groups()
        if left:
            start = int(left)
            end = int(right) if right else size - 1
        else:
            suffix_len = int(right or "0")
            start = max(0, size - suffix_len)
            end = size - 1
        if start >= size or end < start:
            return StreamingResponse(iter(()), status_code=416, media_type=media_type, headers={**headers, "Content-Range": f"bytes */{size}"})
        end = min(end, size - 1)
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = max(0, end - start + 1)
    headers["Content-Length"] = str(length)
    def chunks():
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk: break
                remaining -= len(chunk)
                yield chunk
    return StreamingResponse(chunks(), status_code=status_code, media_type=media_type, headers=headers)


class ForecastRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 128:
            raise ValueError("Tối đa 128 đặc trưng cho mỗi yêu cầu")
        cleaned: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not key.strip() or len(key) > 160:
                raise ValueError("Tên đặc trưng phải là chuỗi 1–160 ký tự")
            number = _finite_number(raw, f"features.{key}", minimum=-1e12, maximum=1e12)
            cleaned[key.strip()] = number
        return cleaned


class ForecastObservationsRequest(BaseModel):
    observations: list[dict[str, Any]] = Field(min_length=1, max_length=50000)
    previous_features: dict[str, float] = Field(default_factory=dict)


class AlertUpdate(BaseModel):
    status: str


class AICCDataVideoRequest(BaseModel):
    path: str


class CameraDataFeedRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)


class AnalyticsTrackPoint(BaseModel):
    """One track point normalized to [0, 1] in its camera frame."""
    track_id: str = Field(min_length=1, max_length=128)
    class_name: str = Field(default="unknown", min_length=1, max_length=64)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnalyticsCountingLine(BaseModel):
    """Counting line endpoints normalized to [0, 1] in camera coordinates."""
    line_id: str = Field(min_length=1, max_length=64)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class RoutePoint(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    lat: float
    lng: float
    camera_id: str | None = Field(default=None, max_length=64)

    @field_validator("lat")
    @classmethod
    def validate_hanoi_latitude(cls, value: float) -> float:
        value = _finite_number(value, "lat", minimum=20.35, maximum=21.65)
        return value

    @field_validator("lng")
    @classmethod
    def validate_hanoi_longitude(cls, value: float) -> float:
        value = _finite_number(value, "lng", minimum=105.0, maximum=106.2)
        return value


class RouteAlternativesRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    analysis_job_id: str | None = Field(default=None, max_length=64)

    @field_validator("analysis_job_id")
    @classmethod
    def validate_analysis_job_id(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("analysis_job_id không hợp lệ")
        return value


class LiveTelemetryRequest(BaseModel):
    camera_id: str = Field(min_length=2, max_length=64)
    source: str = Field(default="rtsp", max_length=16)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ci: float = Field(ge=0, le=1)
    speed_kmh: float | None = Field(default=None, ge=0, le=250)
    flow_pcu: float | None = Field(default=None, ge=0, le=1000000)

    @field_validator("source")
    @classmethod
    def validate_live_source(cls, value: str) -> str:
        if value != "rtsp":
            raise ValueError("Chỉ chấp nhận telemetry từ collector đã xác thực là RTSP")
        return value

    @field_validator("observed_at")
    @classmethod
    def ensure_telemetry_timestamp_aware(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class AnalyticsFrameRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tracks: list[AnalyticsTrackPoint] = Field(default_factory=list, max_length=1000)
    counting_lines: list[AnalyticsCountingLine] = Field(default_factory=list, max_length=50)
    detector_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("timestamp")
    @classmethod
    def ensure_aware_timestamp(cls, value: datetime) -> datetime:
        # Treat caller-supplied naive timestamps as UTC, never as the server's local timezone.
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# One bounded-memory analytics engine per configured camera. It consumes tracker output only;
# no endpoint in this module claims to capture RTSP or perform detection by itself.
ANALYTICS_ENGINES: dict[str, TrafficAnalyticsEngine] = {}
ANALYTICS_ENGINES_LOCK = threading.RLock()


# One CPU-heavy AICC job at a time; progress is polled independently by the browser.
AICC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="traffic-inference")
AICC_JOB_LOCK = threading.RLock()
AICC_JOBS: dict[str, dict[str, Any]] = {}
AICC_MAX_STORED_JOBS = 12
# Video review shares the single CPU inference queue with the legacy AICC job pipeline.
# This prevents two detector jobs from saturating the CPU at the same time.
VIDEO_REVIEW_LOCK = AICC_JOB_LOCK
VIDEO_REVIEW_JOBS: dict[str, dict[str, Any]] = {}
VIDEO_REVIEW_MAX_STORED_JOBS = 8
VIDEO_REVIEW_RETENTION_SECONDS = 3600
LIVE_TELEMETRY: dict[str, dict[str, Any]] = {}
LIVE_TELEMETRY_LOCK = threading.RLock()
LIVE_TELEMETRY_HISTORY: dict[str, list[dict[str, Any]]] = {}
LIVE_TELEMETRY_HISTORY_MAX_AGE_SECONDS = 900
LIVE_TELEMETRY_HISTORY_MAX_SAMPLES = 60
MAX_LIVE_TELEMETRY_CAMERAS = 512


def _recent_live_telemetry() -> dict[str, dict[str, Any]]:
    """Return only fresh observations emitted by the authorized RTSP collector contract."""
    now = datetime.now(timezone.utc).timestamp()
    with LIVE_TELEMETRY_LOCK:
        snapshot = {key: dict(value) for key, value in LIVE_TELEMETRY.items()}
    return {
        camera_id: record for camera_id, record in snapshot.items()
        if 0 <= now - float(record.get("observed_epoch", 0)) <= 180
    }


def _public_camera_rows() -> list[dict[str, Any]]:
    """Return camera metadata without presenting configured fixture metrics as live readings."""
    telemetry = _recent_live_telemetry()
    rows: list[dict[str, Any]] = []
    for configured in CAMERAS:
        camera_id = str(configured["id"])
        record = telemetry.get(camera_id)
        row = dict(configured)
        row["feed"] = _camera_feed_info(camera_id)
        row["telemetry_live"] = bool(record)
        if record:
            row.update({
                "status": "online",
                "ci": float(record["ci"]),
                "speed": record.get("speed_kmh"),
                "vehicles": None,
                "occupancy": None,
                "updated": str(record.get("observed_at", "Telemetry mới")),
                "telemetry_source": "authorized_rtsp_collector",
            })
        else:
            row.update({
                "status": "unverified",
                "ci": None,
                "speed": None,
                "vehicles": None,
                "occupancy": None,
                "updated": "Chưa có telemetry",
                "telemetry_source": None,
            })
        rows.append(row)
    return rows


def _prune_aicc_jobs() -> None:
    completed = [(job_id, job.get("finished_epoch", 0)) for job_id, job in AICC_JOBS.items() if job.get("status") in {"succeeded", "failed"}]
    completed.sort(key=lambda row: row[1])
    while len(AICC_JOBS) > AICC_MAX_STORED_JOBS and completed:
        job_id, _ = completed.pop(0)
        AICC_JOBS.pop(job_id, None)


def _active_aicc_job_exists() -> bool:
    with AICC_JOB_LOCK:
        return (any(job.get("status") in {"queued", "running"} for job in AICC_JOBS.values())
                or any(job.get("status") in {"queued", "running"} for job in VIDEO_REVIEW_JOBS.values()))


def _run_aicc_job(job_id: str, video_path: Path, filename: str, source_path: str | None, cleanup: bool) -> None:
    try:
        with AICC_JOB_LOCK:
            AICC_JOBS[job_id]["status"] = "running"
            AICC_JOBS[job_id]["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        bundle = aicc_runtime.load()
        def on_progress(processed: int, total: int) -> None:
            percent = int(min(99, max(0, processed / max(1, total) * 100)))
            with AICC_JOB_LOCK:
                job = AICC_JOBS.get(job_id)
                if job and job.get("status") == "running":
                    job["progress_percent"] = percent
                    job["frames_decoded"] = processed
                    job["frames_total"] = total
        result = process_video(video_path, bundle, progress_callback=on_progress)
        result["filename"] = filename
        if source_path:
            result["source_path"] = source_path
        with AICC_JOB_LOCK:
            job = AICC_JOBS[job_id]
            job.update({"status":"succeeded", "progress_percent":100, "result":result})
    except Exception as exc:
        logger.exception("AICC job %s failed", job_id)
        with AICC_JOB_LOCK:
            job = AICC_JOBS.get(job_id)
            if job is not None:
                job.update({"status":"failed", "error":f"{type(exc).__name__}: {str(exc)[:400]}"})
    finally:
        if cleanup:
            try: video_path.unlink(missing_ok=True)
            except OSError: logger.warning("Không xóa được file tạm %s", video_path)
        with AICC_JOB_LOCK:
            if job_id in AICC_JOBS:
                AICC_JOBS[job_id]["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                AICC_JOBS[job_id]["finished_epoch"] = __import__("time").time()
                _prune_aicc_jobs()


def _enqueue_aicc_job(video_path: Path, filename: str, source_path: str | None = None, cleanup: bool = False) -> dict[str, Any]:
    with AICC_JOB_LOCK:
        if _active_aicc_job_exists():
            raise HTTPException(status_code=409, detail="Đang có một lượt AICC xử lý video. Vui lòng chờ lượt đó hoàn tất.")
        job_id = uuid.uuid4().hex
        AICC_JOBS[job_id] = {"job_id":job_id,"status":"queued","filename":filename,"progress_percent":0,"frames_decoded":0,"frames_total":0,"created_at":datetime.now().astimezone().isoformat(timespec="seconds")}
        _prune_aicc_jobs()
        AICC_EXECUTOR.submit(_run_aicc_job, job_id, video_path, filename, source_path, cleanup)
        return {"job_id":job_id,"status":"queued","filename":filename,"poll_url":f"/api/aicc/jobs/{job_id}"}


def _prune_video_review_jobs() -> None:
    now = __import__("time").time()
    completed = [(job_id, job.get("finished_epoch", now)) for job_id, job in VIDEO_REVIEW_JOBS.items()
                 if job.get("status") in {"succeeded", "failed"}]
    completed.sort(key=lambda row: row[1])
    for job_id, finished in completed:
        if now - finished >= VIDEO_REVIEW_RETENTION_SECONDS or len(VIDEO_REVIEW_JOBS) > VIDEO_REVIEW_MAX_STORED_JOBS:
            VIDEO_REVIEW_JOBS.pop(job_id, None)
            shutil.rmtree(VIDEO_REVIEW_ROOT / job_id, ignore_errors=True)


def _cleanup_video_reviews() -> None:
    with VIDEO_REVIEW_LOCK:
        _prune_video_review_jobs()
        now = __import__("time").time()
        for upload in VIDEO_REVIEW_UPLOAD_ROOT.glob("traffic_review_*"):
            if upload.is_file() and upload not in ACTIVE_REVIEW_UPLOADS and now - upload.stat().st_mtime >= VIDEO_REVIEW_RETENTION_SECONDS:
                upload.unlink(missing_ok=True)
        # Remove only our generated UUID directories left behind by a previous process.
        for folder in VIDEO_REVIEW_ROOT.iterdir():
            if (re.fullmatch(r"[a-f0-9]{32}", folder.name) and folder.is_dir()
                    and folder.name not in VIDEO_REVIEW_JOBS
                    and now - folder.stat().st_mtime >= VIDEO_REVIEW_RETENTION_SECONDS):
                shutil.rmtree(folder, ignore_errors=True)


def _run_video_review_job(job_id: str, video_path: Path, filename: str, camera_id: str | None,
                          preserve_real_timing: bool, cleanup: bool) -> None:
    try:
        with VIDEO_REVIEW_LOCK:
            job = VIDEO_REVIEW_JOBS[job_id]
            job.update({"status": "running", "started_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        _load_video_review_models()
        bundle = aicc_runtime.load()
        output_dir = VIDEO_REVIEW_ROOT / job_id
        def on_progress(percent: int, processed: int, total: int) -> None:
            with VIDEO_REVIEW_LOCK:
                active = VIDEO_REVIEW_JOBS.get(job_id)
                if active and active.get("status") == "running":
                    active["progress_percent"] = max(0, min(99, int(percent)))
                    active["frames_decoded"] = int(processed)
                    active["frames_total"] = int(total)
        result = analyze_video(video_path, bundle, output_dir, preserve_real_timing=preserve_real_timing,
                               sample_fps=float(os.getenv("VIDEO_REVIEW_SAMPLE_FPS", "1.0")),
                               progress_callback=on_progress)
        result["filename"] = filename
        result["source_camera_id"] = camera_id
        result["frames"] = [{**frame, "url": f"/api/scenarios/video-reviews/{job_id}/frames/{frame['filename']}"}
                             for frame in result.get("frames", [])]
        result["evidence_csv_url"] = f"/api/scenarios/video-reviews/{job_id}/csv"
        with VIDEO_REVIEW_LOCK:
            active = VIDEO_REVIEW_JOBS.get(job_id)
            if active is not None:
                active.update({"status": "succeeded", "progress_percent": 100, "result": result})
    except Exception as exc:
        logger.exception("Video review job %s failed", job_id)
        with VIDEO_REVIEW_LOCK:
            active = VIDEO_REVIEW_JOBS.get(job_id)
            if active is not None:
                active.update({"status": "failed", "error": str(exc)[:400] if isinstance(exc, VideoReviewError) else "Không thể hoàn tất phân tích video. Vui lòng thử lại hoặc chọn clip khác."})
    finally:
        if cleanup:
            try:
                video_path.unlink(missing_ok=True)
                with VIDEO_REVIEW_LOCK:
                    ACTIVE_REVIEW_UPLOADS.discard(video_path)
            except OSError:
                logger.warning("Không xóa được file video tạm %s", video_path)
        with VIDEO_REVIEW_LOCK:
            active = VIDEO_REVIEW_JOBS.get(job_id)
            if active is not None:
                active["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                active["finished_epoch"] = __import__("time").time()
            _prune_video_review_jobs()


def _load_video_review_models() -> None:
    status = aicc_runtime.status()
    if status.get("missing_files"):
        raise HTTPException(status_code=503, detail="Thiếu mô hình AICC: " + ", ".join(status["missing_files"]))
    try:
        aicc_runtime.load()
        hanoi_video_forecaster.load()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không nạp được mô hình cần cho phân tích video: {str(exc)[:300]}",
        ) from exc


def _review_future_done(future, job_id: str, video_path: Path, cleanup: bool) -> None:
    if not future.cancelled() and future.exception() is None:
        return
    logger.error("Video review worker stopped before completing job %s", job_id)
    with VIDEO_REVIEW_LOCK:
        job = VIDEO_REVIEW_JOBS.get(job_id)
        if job and job.get("status") in {"queued", "running"}:
            job.update(status="failed", error="Tác vụ bị gián đoạn. Vui lòng thử lại.",
                       finished_epoch=__import__("time").time(),
                       finished_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    if cleanup:
        video_path.unlink(missing_ok=True)
        with VIDEO_REVIEW_LOCK:
            ACTIVE_REVIEW_UPLOADS.discard(video_path)


def _enqueue_video_review_job(video_path: Path, filename: str, camera_id: str | None,
                              preserve_real_timing: bool, cleanup: bool) -> dict[str, Any]:
    # Model loading belongs to the worker, never to the upload request/event loop.
    with VIDEO_REVIEW_LOCK:
        if _active_aicc_job_exists():
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        job_id = uuid.uuid4().hex
        VIDEO_REVIEW_JOBS[job_id] = {"job_id": job_id, "status": "queued", "filename": filename,
            "camera_id": camera_id, "preserve_real_timing": preserve_real_timing, "progress_percent": 0,
            "frames_decoded": 0, "frames_total": 0,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        _prune_video_review_jobs()
        try:
            future = AICC_EXECUTOR.submit(_run_video_review_job, job_id, video_path, filename, camera_id, preserve_real_timing, cleanup)
            if hasattr(future, "add_done_callback"):
                future.add_done_callback(lambda done: _review_future_done(done, job_id, video_path, cleanup))
        except Exception as exc:
            # A rejected submission must not leave a queued job blocking every retry.
            VIDEO_REVIEW_JOBS[job_id].update({"status": "failed", "error": "Không khởi tạo được worker CPU.",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "finished_epoch": __import__("time").time()})
            logger.exception("Video review submission %s failed", job_id)
            raise HTTPException(status_code=503, detail="Hệ thống chưa thể bắt đầu phân tích. Vui lòng thử lại sau.") from exc
        return {"job_id": job_id, "status": "queued", "filename": filename,
                "poll_url": f"/api/scenarios/video-reviews/{job_id}"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/map/config")
def map_config() -> dict[str, Any]:
    """Return safe browser map configuration. Traffic tile credentials must be browser-restricted keys."""
    tile_url = os.getenv("MAP_TILE_URL_TEMPLATE", "https://tile.openstreetmap.org/{z}/{x}/{y}.png").strip()
    base_name = os.getenv("MAP_BASE_NAME", "OpenStreetMap").strip()[:80] or "OpenStreetMap"
    base_attribution = os.getenv("MAP_ATTRIBUTION", '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>').strip()
    traffic_url = os.getenv("TRAFFIC_TILE_URL_TEMPLATE", "").strip()
    traffic_attribution = os.getenv("TRAFFIC_TILE_ATTRIBUTION", "").strip()[:300]
    traffic_name = os.getenv("TRAFFIC_LAYER_NAME", "Giao thông trực tiếp").strip()[:80] or "Giao thông trực tiếp"
    warning = None

    def valid_template(value: str, name: str) -> bool:
        nonlocal warning
        if not value:
            return False
        parsed = urlsplit(value.replace("{z}", "0").replace("{x}", "0").replace("{y}", "0").replace("{s}", "a"))
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}):
            warning = f"{name} bị tắt vì chỉ chấp nhận HTTPS (HTTP chỉ cho localhost)."
            return False
        if "{z}" not in value or "{x}" not in value or "{y}" not in value:
            warning = f"{name} bị tắt vì URL phải có placeholder {{z}}, {{x}}, {{y}}."
            return False
        return True

    if not valid_template(tile_url, "MAP_TILE_URL_TEMPLATE"):
        tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        base_name = "OpenStreetMap"
        base_attribution = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>'
    traffic_enabled = valid_template(traffic_url, "TRAFFIC_TILE_URL_TEMPLATE") if traffic_url else False
    if not traffic_enabled:
        traffic_url = ""
        traffic_attribution = ""
    try:
        refresh_seconds = max(10, min(120, int(os.getenv("CAMERA_REFRESH_SECONDS", "15"))))
    except ValueError:
        refresh_seconds = 15
    return {
        "base_map_name": base_name,
        "tile_url_template": tile_url,
        "base_attribution": base_attribution,
        "traffic_layer_enabled": traffic_enabled,
        "traffic_layer_name": traffic_name,
        "traffic_tile_url_template": traffic_url,
        "traffic_attribution": traffic_attribution,
        "camera_refresh_seconds": refresh_seconds,
        "traffic_source_note": "Lớp giao thông chỉ hoạt động khi người vận hành cấu hình tile/API được cấp phép.",
        "configuration_warning": warning,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "build_commit": os.getenv("RENDER_GIT_COMMIT"), "service": "hanoi-traffic-forecasting", "time": datetime.now().astimezone().isoformat(timespec="seconds")}


# WEATHER_TRAFFIC_TERMINAL_V1
@app.get("/api/weather")
def weather_forecast(latitude: float = 21.0285, longitude: float = 105.8542) -> dict[str, Any]:
    """Current conditions and hourly weather forecast for the selected Hanoi coordinate."""
    try:
        return get_weather_report(latitude, longitude)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail="Nguồn thời tiết Open-Meteo hiện không khả dụng; thử lại sau.") from exc

@app.get("/api/system")
def system_status() -> dict[str, Any]:
    status = runtime.status()
    telemetry = _recent_live_telemetry()
    aicc_status = aicc_runtime.status()
    hanoi_status = hanoi_video_forecaster.status()
    public_upload_limit = (
        MAX_PUBLIC_VIDEO_REVIEW_BYTES if PUBLIC_DEMO_MODE else MAX_VIDEO_REVIEW_BYTES
    )
    public_duration_limit = (
        MAX_PUBLIC_VIDEO_REVIEW_DURATION_SECONDS
        if PUBLIC_DEMO_MODE
        else MAX_VIDEO_REVIEW_DURATION_SECONDS
    )
    return {
        "service": "Đang hoạt động",
        "model_mode": status["mode"],
        "model_status": status,
        "forecast_provider": forecast_providers.status(),
        "camera_total": len(CAMERAS),
        "camera_online": len(telemetry),
        "alerts_open": 0,
        "cpu_target": "CPU · ONNX Runtime",
        "public_demo_mode": PUBLIC_DEMO_MODE,
        "video_review": video_review_access(),
        "process_id": os.getpid(),
        "launch_instance": os.getenv("HANOI_LAUNCH_INSTANCE") if not PUBLIC_DEMO_MODE else None,
        "video_review_upload_enabled": video_review_upload_enabled(),
        "video_review_ready": (
            video_review_upload_enabled()
            and aicc_status.get("ready", False)
            and hanoi_status.get("ready", False)
        ),
        "video_review_upload_limit_mb": public_upload_limit // (1024 * 1024),
        "video_review_max_duration_seconds": public_duration_limit,
        "camera_config_warning": CAMERA_CONFIG_WARNING,
        "telemetry_source": "authorized_rtsp_collector" if telemetry else "not_connected",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    """Expose live collector telemetry when present; never forecast from static sample values."""
    now = datetime.now().astimezone()
    telemetry = _recent_live_telemetry()
    rows = _public_camera_rows()
    speed_values = [float(item["speed_kmh"]) for item in telemetry.values()
                    if item.get("speed_kmh") is not None and math.isfinite(float(item["speed_kmh"]))]
    ci_values = [float(item["ci"]) for item in telemetry.values()
                 if item.get("ci") is not None and math.isfinite(float(item["ci"]))]
    data_source = "live_telemetry" if telemetry else "reference_fixture"
    # A fresh single snapshot is not a time series. Even with an ONNX file installed, only
    # /api/forecast/from-observations may infer from an explicitly supplied measured feature set.
    forecast = {
        "mode": "unavailable",
        "provider": forecast_providers.status().get("provider", "local_onnx"),
        "predictions": [],
        "reason": "no_observation_series",
        "message": "Dashboard does not create forecasts from fixture values or one isolated telemetry snapshot.",
    }
    return {
        "mode": "demo",
        "data_source": data_source,
        "generated_at": now.isoformat(timespec="seconds"),
        "summary": {
            "camera_total": len(CAMERAS),
            "camera_online": len(telemetry),
            "total_vehicles": None,
            "average_speed": round(sum(speed_values) / len(speed_values), 1) if speed_values else None,
            "critical_points": sum(1 for value in ci_values if value >= 0.78) if ci_values else None,
            "average_ci": round(sum(ci_values) / len(ci_values), 3) if ci_values else None,
        },
        "forecast": forecast,
        "cameras": rows,
        "alerts": [],
        "trend": [],
        "telemetry_freshness_seconds": 180,
    }


@app.get("/api/cameras")
def get_cameras() -> dict[str, Any]:
    """Return configured camera reference points plus genuine collector telemetry, if any."""
    global CAMERAS, CAMERA_CONFIG_WARNING
    CAMERAS, CAMERA_CONFIG_WARNING = _load_cameras()
    telemetry = _recent_live_telemetry()
    try:
        refresh_seconds = max(10, min(120, int(os.getenv("CAMERA_REFRESH_SECONDS", "15"))))
    except ValueError:
        refresh_seconds = 15
    return {
        "items": _public_camera_rows(),
        "total": len(CAMERAS),
        "demo": not bool(telemetry),
        "telemetry_mode": "rtsp_collector" if telemetry else "not_connected",
        "metadata_source": "data/cameras.json",
        "camera_refresh_seconds": refresh_seconds,
        "camera_config_warning": CAMERA_CONFIG_WARNING,
        "fresh_telemetry_camera_count": len(telemetry),
    }


@app.get("/api/cameras/{camera_id}/feed-info")
def get_camera_feed_info(camera_id: str) -> dict[str, Any]:
    _assert_camera_exists(camera_id)
    return {"camera_id": camera_id, **_camera_feed_info(camera_id), "upload_enabled": not PUBLIC_DEMO_MODE}


@app.get("/api/cameras/{camera_id}/feed")
def get_camera_feed(camera_id: str, request: Request) -> Response:
    _assert_camera_exists(camera_id)
    path, _source_type = _camera_feed_path(camera_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Camera này chưa được gán clip. Hãy gán video riêng từ máy tính hoặc từ thư mục data/.")
    return _video_range_response(request, path)


@app.post("/api/cameras/{camera_id}/feed")
async def upload_camera_feed(camera_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Attach a validated local video clip to a configurable camera point."""
    require_write_access()
    _assert_camera_exists(camera_id)
    original_name = Path(file.filename or "camera.mp4").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video hỗ trợ: mp4, avi, mkv, mov, mpeg, mpg, webm")
    CAMERA_FEED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(prefix=f"{camera_id}_", suffix=suffix, dir=CAMERA_FEED_DIR, delete=False) as stream:
            tmp_name = stream.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                total += len(chunk)
                if total > MAX_CAMERA_FEED_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video vượt giới hạn {MAX_CAMERA_FEED_BYTES // (1024*1024)} MB")
                stream.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Tệp video rỗng")
        cap = cv2.VideoCapture(tmp_name)
        try:
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Không mở được video; kiểm tra codec/định dạng")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            ok, _frame = cap.read()
            if not ok or not math.isfinite(fps) or fps <= 0:
                raise HTTPException(status_code=400, detail="Video không có frame đọc được hoặc metadata FPS hợp lệ")
        finally:
            cap.release()
        final_name = f"{camera_id}_{uuid.uuid4().hex[:10]}{suffix}"
        dest = (CAMERA_FEED_DIR / final_name).resolve()
        dest.relative_to(CAMERA_FEED_DIR.resolve())
        previous = _read_camera_feed_index().get(camera_id)
        os.replace(tmp_name, dest)
        tmp_name = None
        index = _read_camera_feed_index()
        index[camera_id] = final_name
        _save_camera_feed_index(index)
        if previous and previous != final_name:
            old = (CAMERA_FEED_DIR / previous).resolve()
            try:
                old.relative_to(CAMERA_FEED_DIR.resolve())
                old.unlink(missing_ok=True)
            except ValueError:
                logger.warning("Bỏ qua đường dẫn video cũ ngoài thư mục hợp lệ")
            except OSError:
                # The browser may still be reading the previous clip on Windows; keep it as an orphan rather than failing a successful replacement.
                logger.warning("Clip cũ đang được sử dụng; giữ lại tệp %s để tránh làm hỏng luồng hiện tại", previous)
        logger.info("Assigned camera clip camera=%s size=%s bytes", camera_id, total)
        return {"ok": True, "camera_id": camera_id, "feed": _camera_feed_info(camera_id), "message": "Đã gán clip cho điểm camera trong bản demo."}
    except HTTPException:
        raise
    except OSError as exc:
        logger.exception("Lỗi ghi video camera %s", camera_id)
        raise HTTPException(status_code=500, detail="Không ghi được file video; kiểm tra quyền ghi và dung lượng đĩa.") from exc
    finally:
        await file.close()
        if tmp_name:
            try: Path(tmp_name).unlink(missing_ok=True)
            except OSError: pass


@app.post("/api/cameras/{camera_id}/feed-from-data")
def assign_camera_feed_from_data(camera_id: str, body: CameraDataFeedRequest) -> dict[str, Any]:
    """Assign a video already stored below data/ to one camera, copying it into managed feed storage."""
    require_write_access()
    _assert_camera_exists(camera_id)
    source = (DATA_DIR / Path(body.path)).resolve()
    try:
        source.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Đường dẫn video phải nằm trong data/.") from exc
    try:
        source.relative_to(CAMERA_FEED_DIR.resolve())
        raise HTTPException(status_code=400, detail="Không chọn video đã nằm trong camera_feeds/ làm nguồn.")
    except ValueError:
        pass
    if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ.")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy video trong data/.")
    size = source.stat().st_size
    if size <= 0:
        raise HTTPException(status_code=400, detail="Video rỗng.")
    if size > MAX_CAMERA_FEED_BYTES:
        raise HTTPException(status_code=413, detail="Video vượt giới hạn 500 MB.")
    capture = cv2.VideoCapture(str(source))
    try:
        ok, _frame = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if not capture.isOpened() or not ok or not math.isfinite(fps) or fps <= 0:
            raise HTTPException(status_code=400, detail="Video không có frame đọc được hoặc thiếu FPS hợp lệ.")
    finally:
        capture.release()
    CAMERA_FEED_DIR.mkdir(parents=True, exist_ok=True)
    final_name = f"{camera_id}_{uuid.uuid4().hex[:10]}{source.suffix.lower()}"
    destination = (CAMERA_FEED_DIR / final_name).resolve()
    destination.relative_to(CAMERA_FEED_DIR.resolve())
    temp_dest = CAMERA_FEED_DIR / f".{final_name}.tmp"
    previous = _read_camera_feed_index().get(camera_id)
    try:
        shutil.copyfile(source, temp_dest)
        os.replace(temp_dest, destination)
        index = _read_camera_feed_index()
        index[camera_id] = final_name
        _save_camera_feed_index(index)
    except OSError as exc:
        temp_dest.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        logger.exception("Không thể gán video từ data/ cho camera %s", camera_id)
        raise HTTPException(status_code=500, detail="Không thể sao chép clip vào camera_feeds/. Kiểm tra dung lượng và quyền ghi.") from exc
    if previous and previous != final_name:
        old = (CAMERA_FEED_DIR / previous).resolve()
        try:
            old.relative_to(CAMERA_FEED_DIR.resolve())
            old.unlink(missing_ok=True)
        except (ValueError, OSError):
            logger.warning("Không xóa được clip cũ %s", previous)
    logger.info("Assigned data clip to camera=%s source=%s", camera_id, source.name)
    return {"ok": True, "camera_id": camera_id, "feed": _camera_feed_info(camera_id), "message": "Đã gán clip trong data/ cho camera."}


@app.delete("/api/cameras/{camera_id}/feed")
def delete_camera_feed(camera_id: str) -> dict[str, Any]:
    require_write_access()
    _assert_camera_exists(camera_id)
    index = _read_camera_feed_index()
    filename = index.pop(camera_id, None)
    _save_camera_feed_index(index)
    if filename:
        path = (CAMERA_FEED_DIR / filename).resolve()
        try:
            path.relative_to(CAMERA_FEED_DIR.resolve())
            path.unlink(missing_ok=True)
        except ValueError:
            logger.warning("Bỏ qua đường dẫn video cũ ngoài thư mục hợp lệ")
    return {"ok": True, "camera_id": camera_id, "feed": _camera_feed_info(camera_id), "message": "Đã gỡ clip đã gán. Video mẫu chung (nếu có) vẫn có thể xuất hiện."}


@app.get("/api/alerts")
def get_alerts() -> dict[str, Any]:
    return {"items": [], "total": 0, "demo": False, "source": "not_connected",
            "message": "Chưa có engine sinh cảnh báo từ telemetry đã đo; không hiển thị cảnh báo giả."}


@app.post("/api/alerts/{alert_id}")
def update_alert(alert_id: str, body: AlertUpdate) -> dict[str, Any]:
    require_write_access()
    allowed = {"Chưa xử lý", "Đang theo dõi", "Đã ghi nhận"}
    if body.status not in allowed:
        raise HTTPException(status_code=400, detail="Trạng thái không hợp lệ")
    for alert in ALERTS:
        if alert["id"] == alert_id:
            alert["status"] = body.status
            return {"ok": True, "item": alert}
    raise HTTPException(status_code=404, detail="Không tìm thấy cảnh báo")


@app.post("/api/forecast")
def forecast(body: ForecastRequest) -> dict[str, Any]:
    """Predict via the configured local ONNX runtime or a contract-compatible HTTP provider."""
    return forecast_providers.predict(body.features)


@app.post("/api/forecast/from-observations")
def forecast_from_observations(body: ForecastObservationsRequest) -> dict[str, Any]:
    """Build manifest-ordered features from tracked observations and route them to the selected provider."""
    try:
        feature_build = aggregate_traffic_features(
            body.observations,
            runtime.manifest,
            previous_features=body.previous_features,
        )
    except FeatureContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    quality = feature_build["quality"]
    if not quality["ready_for_real_forecast"]:
        return {
            "mode": "insufficient_features",
            "provider": forecast_providers.status().get("provider"),
            "features": feature_build["features"],
            "feature_quality": quality,
            "predictions": [],
            "message": quality["warning"] or "Chưa đủ đặc trưng đo đạc để dự báo.",
        }
    result = forecast_providers.predict(feature_build["features"])
    return {
        "mode": result.get("mode", "error"),
        "features": feature_build["features"],
        "feature_quality": quality,
        "forecast": result,
    }


@app.get("/api/providers/status")
def forecast_provider_status() -> dict[str, Any]:
    """Expose safe provider metadata; never expose API keys or authorization headers."""
    return forecast_providers.status()


@app.get("/api/models/status")
def model_status() -> dict[str, Any]:
    runtime_status = runtime.status()
    provider_status = forecast_providers.status()
    runtime_status["forecast_provider"] = provider_status
    runtime_status["forecast_provider_name"] = provider_status.get("provider", "local_onnx")
    if provider_status.get("mode") == "remote_api":
        runtime_status["mode"] = "production"
    elif provider_status.get("provider") == "remote_http" and not provider_status.get("configured"):
        runtime_status["mode"] = "demo"
    return runtime_status


@app.get("/api/models/manifest")
def read_manifest() -> dict[str, Any]:
    return runtime.manifest


@app.post("/api/models/reload")
def reload_models() -> dict[str, Any]:
    require_write_access()
    try:
        status = runtime.reload()
        return {"ok": True, "status": status}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/upload")
async def upload_artifact(kind: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Stream and validate a model artifact rather than reading an unbounded file into RAM."""
    require_write_access()
    allowed = {"detector": "detector.onnx", "forecaster": "forecaster.onnx", "manifest": "manifest.json"}
    if kind not in allowed:
        raise HTTPException(status_code=400, detail="kind phải là detector, forecaster hoặc manifest")
    suffix = Path(file.filename or "").suffix.lower()
    if kind in {"detector", "forecaster"} and suffix != ".onnx":
        raise HTTPException(status_code=400, detail="Mô hình phải ở định dạng .onnx")
    if kind == "manifest" and suffix != ".json":
        raise HTTPException(status_code=400, detail="Manifest phải ở định dạng .json")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="artifact_", suffix=suffix, dir=ARTIFACT_DIR, delete=False) as tmp:
            temp_path = Path(tmp.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                total += len(chunk)
                if total > MAX_ARTIFACT_BYTES:
                    raise HTTPException(status_code=413, detail=f"Artifact vượt giới hạn {MAX_ARTIFACT_BYTES // (1024*1024)} MB")
                tmp.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Tệp rỗng")
        if kind == "manifest":
            try:
                parsed = json.loads(temp_path.read_text(encoding="utf-8-sig"))
                normalized = _normalize_manifest_payload(parsed)
                temp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"manifest.json không hợp lệ: {exc}") from exc
        elif kind in {"detector", "forecaster"}:
            # Validate the ONNX graph before replacing a known-good artifact. This prevents a renamed
            # HTML/ZIP or broken ONNX file from being accepted as an active model.
            try:
                import onnxruntime as ort  # type: ignore
                session = ort.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
                if not session.get_inputs() or not session.get_outputs():
                    raise ValueError("ONNX cần có ít nhất một input và một output")
                if kind == "forecaster":
                    output_shapes = [getattr(meta, "shape", None) for meta in session.get_outputs()]
                    known_sizes = []
                    for shape in output_shapes:
                        if isinstance(shape, (list, tuple)):
                            known_sizes.extend(dim for dim in shape if isinstance(dim, int))
                    if output_shapes and all(isinstance(shape, (list, tuple)) for shape in output_shapes):
                        if sum(1 for shape in output_shapes if shape is not None) == 1 and known_sizes and math.prod(known_sizes) < 3:
                            raise ValueError("Forecaster ONNX phải có khả năng trả về ít nhất 3 giá trị CI")
            except ImportError as exc:
                raise HTTPException(status_code=503, detail="onnxruntime chưa được cài; không thể xác thực model trước khi thay thế") from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"ONNX không hợp lệ cho vai trò {kind}: {str(exc)[:220]}") from exc
        # Keep previous file intact until the new payload has been streamed and validated.
        dest = ARTIFACT_DIR / allowed[kind]
        os.replace(temp_path, dest)
        temp_path = None
        return {"ok": True, "filename": allowed[kind], "bytes": total, "message": f"Đã lưu {allowed[kind]}. Hãy nạp lại mô hình để áp dụng."}
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Không lưu được artifact; kiểm tra quyền ghi và dung lượng đĩa.") from exc
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


@app.post("/api/models/manifest")
def save_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    require_write_access()
    try:
        normalized = _normalize_manifest_payload(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Manifest không hợp lệ: {exc}") from exc
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="manifest_", dir=MANIFEST_PATH.parent, delete=False) as stream:
            tmp_path = Path(stream.name)
            json.dump(normalized, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp_path, MANIFEST_PATH)
        tmp_path = None
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Không lưu được manifest; kiểm tra quyền ghi và dung lượng đĩa.") from exc
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
    return {"ok": True, "manifest": normalized, "message": "Đã lưu manifest hợp lệ. Nạp lại mô hình để áp dụng."}


@app.get("/api/models/download-manifest")
def download_manifest() -> FileResponse:
    return FileResponse(MANIFEST_PATH, filename="manifest.json", media_type="application/json")


@app.get("/api/aicc/status")
def aicc_status() -> dict[str, Any]:
    """Report readiness of the separate AICC sequence-classification bundle."""
    status = aicc_runtime.status()
    status.update(video_review_access())
    hanoi_status = hanoi_video_forecaster.status()
    status["hanoi_forecaster"] = hanoi_status
    status["video_review_enabled"] = (
        video_review_upload_enabled()
        and status.get("artifacts_present", False)
        and hanoi_status.get("artifacts_present", False)
    )
    status["video_review_ready"] = (
        status["video_review_enabled"]
        and status.get("ready", False)
        and hanoi_status.get("ready", False)
    )
    status["upstream"] = "https://github.com/dinoraptor999/AICC"
    status["upload_limit_mb"] = (
        MAX_PUBLIC_VIDEO_REVIEW_BYTES if PUBLIC_DEMO_MODE else MAX_VIDEO_REVIEW_BYTES
    ) // (1024 * 1024)
    status["upload_max_duration_seconds"] = (
        MAX_PUBLIC_VIDEO_REVIEW_DURATION_SECONDS
        if PUBLIC_DEMO_MODE
        else MAX_VIDEO_REVIEW_DURATION_SECONDS
    )
    if PUBLIC_DEMO_MODE:
        if status["video_review_enabled"]:
            status["message"] = "Có thể tải video lên để phân tích clip."
        elif not PUBLIC_VIDEO_REVIEW_ENABLED:
            status["message"] = "Tải video phân tích đang bị tắt ở public demo."
        else:
            missing = status.get("missing_files", []) + hanoi_status.get("missing_files", [])
            detail = ", ".join(missing) or status.get("error") or hanoi_status.get("error")
            status["message"] = (
                "Thiếu hoặc không nạp được model artifacts: " + str(detail)
                if detail
                else "Model chưa được nạp; tải video để khởi tạo trước."
            )
    elif not AICC_VIDEO_ENABLED:
        status["message"] = "AICC_VIDEO_ENABLED=false; bật biến môi trường để chạy phân tích video."
    elif status.get("missing_files"):
        status["message"] = "Cần đặt detector.onnx, predictor.onnx và scaler.pkl vào models/ theo manifest AICC."
    elif hanoi_status.get("missing_files"):
        status["message"] = "Thiếu bộ model Hanoi cho video review; kiểm tra hanoi_predictor.onnx, hanoi_scaler.pkl và hanoi_manifest.json trong models/."
    elif not status.get("loaded"):
        status["message"] = status.get("error") or "Artifacts đã có; model sẽ được nạp lần đầu khi gửi video."
    else:
        status["message"] = "AICC model bundle đã được nạp trên CPU."
    return status


@app.post("/api/aicc/load")
def aicc_load_models() -> dict[str, Any]:
    """Explicitly validate/load the AICC bundle when the operator presses 'Check model'."""
    if PUBLIC_DEMO_MODE or not AICC_VIDEO_ENABLED:
        raise HTTPException(status_code=403, detail="Nạp AICC chỉ bật ở chế độ local.")
    status = aicc_runtime.status()
    if status.get("missing_files"):
        raise HTTPException(status_code=503, detail="Thiếu artifacts: " + ", ".join(status["missing_files"]))
    try:
        bundle = aicc_runtime.load()
        return {"ok": True, "status": aicc_runtime.status(), "model_version": bundle["manifest"].get("version")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Không nạp được AICC: {str(exc)[:300]}") from exc


@app.get("/api/aicc/jobs/{job_id}")
def aicc_job_status(job_id: str) -> dict[str, Any]:
    with AICC_JOB_LOCK:
        job = AICC_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ AICC hoặc tác vụ đã hết hạn")
        payload = {key: value for key, value in job.items() if key not in {"result", "finished_epoch"}}
        if job.get("status") == "succeeded": payload["result"] = job.get("result")
        return payload


@app.get("/api/aicc/data-videos")
def aicc_data_videos() -> dict[str, Any]:
    """List locally extracted videos under data/; no file path outside that folder is exposed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        try:
            if any(path.resolve().is_relative_to(folder.resolve()) for folder in (CAMERA_FEED_DIR, VIDEO_REVIEW_ROOT, VIDEO_REVIEW_UPLOAD_ROOT)):
                continue
            size = path.stat().st_size
            if not 0 < size <= MAX_VIDEO_REVIEW_BYTES:
                continue
            with path.open("rb") as video:
                video.read(1)  # Listing promises an accessible file, not codec/model readiness.
            relative = path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
            items.append({"path": relative, "name": path.name, "size_mb": round(size / (1024 * 1024), 2)})
        except (OSError, ValueError):
            continue
        if len(items) >= 200:
            break
    return {"items": items, "total": len(items), "data_dir": "data/"}


@app.post("/api/aicc/predict-data-video", status_code=202)
def aicc_predict_data_video(body: AICCDataVideoRequest) -> dict[str, Any]:
    """Queue a local clip for background CPU inference and return a job id immediately."""
    if PUBLIC_DEMO_MODE or not AICC_VIDEO_ENABLED:
        raise HTTPException(status_code=403, detail="Phân tích video AICC chỉ bật ở chế độ local.")
    requested = Path(body.path)
    video_path = (DATA_DIR / requested).resolve()
    if not video_path.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Đường dẫn video không hợp lệ")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy video trong data/")
    if video_path.stat().st_size > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail=f"Video vượt giới hạn {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    return _enqueue_aicc_job(video_path, video_path.name, video_path.relative_to(DATA_DIR.resolve()).as_posix(), cleanup=False)


@app.post("/api/aicc/predict-video", status_code=202)
async def aicc_predict_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Stream an uploaded video to disk, then process asynchronously without buffering it all in RAM."""
    if PUBLIC_DEMO_MODE or not AICC_VIDEO_ENABLED:
        raise HTTPException(status_code=403, detail="Phân tích video AICC chỉ bật ở chế độ local; public demo khóa tác vụ CPU nặng.")
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video hỗ trợ: mp4, avi, mkv, mov, mpeg, mpg, webm")
    temp_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="aicc_video_", delete=False) as tmp:
            temp_path = Path(tmp.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video vượt giới hạn {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
                tmp.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Tệp video rỗng")
        # Reject non-video files before a worker slot is consumed.
        cap = cv2.VideoCapture(str(temp_path))
        try:
            ok, _frame = cap.read()
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            if not cap.isOpened() or not ok or not math.isfinite(fps) or fps <= 0:
                raise HTTPException(status_code=400, detail="Tệp tải lên không phải video đọc được hoặc thiếu FPS hợp lệ")
        finally:
            cap.release()
        job = _enqueue_aicc_job(temp_path, filename, source_path=None, cleanup=True)
        temp_path = None
        return job
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Không lưu được video tạm; kiểm tra dung lượng đĩa.") from exc
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


def _validate_review_upload(tmp_path: Path) -> None:
    cap = cv2.VideoCapture(str(tmp_path))
    try:
        ok, _frame = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not cap.isOpened() or not ok or not math.isfinite(fps) or fps <= 0 or not math.isfinite(frame_count) or frame_count <= 0:
            raise HTTPException(status_code=400, detail="Không đọc được video hoặc FPS không hợp lệ")
        duration = frame_count / fps
        max_duration = (
            MAX_PUBLIC_VIDEO_REVIEW_DURATION_SECONDS
            if PUBLIC_DEMO_MODE
            else MAX_VIDEO_REVIEW_DURATION_SECONDS
        )
        if duration > max_duration:
            raise HTTPException(
                status_code=413,
                detail=f"Video vượt giới hạn {max_duration // 60} phút",
            )
    finally:
        cap.release()


@app.post("/api/scenarios/video-review", status_code=202)
async def create_video_review(
    file: UploadFile = File(...),
    camera_id: str | None = Form(default=None),
    preserve_real_timing: bool = Form(default=False),
) -> dict[str, Any]:
    """Analyze an uploaded clip in the background and return selected frames + explainable state labels."""
    if not video_review_upload_enabled():
        raise HTTPException(status_code=403, detail="Tính năng tải video để phân tích hiện đang bị tắt.")
    if camera_id:
        _assert_camera_exists(camera_id)
    filename = Path(file.filename or "traffic.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video hỗ trợ: mp4, avi, mkv, mov, mpeg, mpg, webm")
    tmp_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="traffic_review_", dir=VIDEO_REVIEW_UPLOAD_ROOT, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with VIDEO_REVIEW_LOCK:
                ACTIVE_REVIEW_UPLOADS.add(tmp_path)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                max_bytes = (
                    MAX_PUBLIC_VIDEO_REVIEW_BYTES if PUBLIC_DEMO_MODE else MAX_VIDEO_REVIEW_BYTES
                )
                if total_bytes > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Video vượt giới hạn {max_bytes // (1024 * 1024)} MB")
                await run_in_threadpool(tmp.write, chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Tệp video rỗng")
        await run_in_threadpool(_validate_review_upload, tmp_path)
        job = _enqueue_video_review_job(tmp_path, filename, camera_id, preserve_real_timing, cleanup=True)
        tmp_path = None
        return job
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Không lưu được video tạm; kiểm tra dung lượng đĩa.") from exc
    finally:
        await file.close()
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
            with VIDEO_REVIEW_LOCK:
                ACTIVE_REVIEW_UPLOADS.discard(tmp_path)


@app.post("/api/scenarios/video-review-from-data", status_code=202)
def create_video_review_from_data(body: dict[str, Any]) -> dict[str, Any]:
    """Analyze a clip already placed under data/ without resending it through the browser."""
    if not AICC_VIDEO_ENABLED or PUBLIC_DEMO_MODE:
        raise HTTPException(status_code=403, detail="Phân tích clip chỉ bật trong môi trường local/protected.")
    raw_path = body.get("path")
    camera_id = body.get("camera_id")
    preserve_real_timing = body.get("preserve_real_timing", False)
    if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > 500:
        raise HTTPException(status_code=422, detail="path trong data/ là bắt buộc")
    if not isinstance(preserve_real_timing, bool):
        raise HTTPException(status_code=422, detail="preserve_real_timing phải là boolean")
    if camera_id is not None:
        if not isinstance(camera_id, str):
            raise HTTPException(status_code=422, detail="camera_id không hợp lệ")
        _assert_camera_exists(camera_id)
    candidate = (DATA_DIR / raw_path).resolve()
    if not candidate.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Đường dẫn video không hợp lệ")
    if candidate.is_relative_to(VIDEO_REVIEW_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Không thể phân tích output đã sinh bởi video review")
    if candidate.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video không còn tồn tại dưới data/. Tải lại danh sách hoặc chọn tệp từ máy tính để upload.")
    if candidate.stat().st_size > MAX_VIDEO_REVIEW_BYTES:
        raise HTTPException(status_code=413, detail=f"Video vượt giới hạn {MAX_VIDEO_REVIEW_BYTES // (1024 * 1024)} MB")
    return _enqueue_video_review_job(candidate, candidate.name, camera_id, preserve_real_timing, cleanup=False)


@app.get("/api/scenarios/video-reviews/{job_id}")
def video_review_status(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ")
    with VIDEO_REVIEW_LOCK:
        _prune_video_review_jobs()
        job = VIDEO_REVIEW_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ hoặc tác vụ đã hết hạn")
        payload = {key: value for key, value in job.items() if key not in {"finished_epoch"}}
        if job.get("status") == "succeeded":
            payload["result"] = job.get("result")
        return payload


def _require_review_artifacts(job_id: str) -> None:
    with VIDEO_REVIEW_LOCK:
        _prune_video_review_jobs()
        if VIDEO_REVIEW_JOBS.get(job_id, {}).get("status") != "succeeded":
            raise HTTPException(status_code=404, detail="Kết quả không còn khả dụng hoặc đã hết hạn.")


@app.get("/api/scenarios/video-reviews/{job_id}/frames/{filename}")
def video_review_frame(job_id: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not re.fullmatch(r"frame_\d{3}\.jpg", filename):
        raise HTTPException(status_code=404, detail="Không tìm thấy frame")
    _require_review_artifacts(job_id)
    target = (VIDEO_REVIEW_ROOT / job_id / "frames" / filename).resolve()
    if not target.is_relative_to((VIDEO_REVIEW_ROOT / job_id / "frames").resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy frame")
    return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/scenarios/video-reviews/{job_id}/csv")
def video_review_csv(job_id: str) -> FileResponse:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ")
    _require_review_artifacts(job_id)
    target = (VIDEO_REVIEW_ROOT / job_id / "evidence.csv").resolve()
    if not target.is_relative_to(VIDEO_REVIEW_ROOT.resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp CSV minh chứng")
    return FileResponse(target, media_type="text/csv", headers={"Cache-Control": "no-store"}, filename=f"hanoi-traffic-evidence-{job_id[:8]}.csv")


@app.post("/api/routes/alternatives")
def route_alternatives(body: RouteAlternativesRequest) -> dict[str, Any]:
    """Fetch real road-network geometries; prefer congestion-aware scoring only with fresh verified RTSP telemetry."""
    if math.hypot((body.origin.lng-body.destination.lng)*111320*math.cos(math.radians(body.origin.lat)),
                  (body.origin.lat-body.destination.lat)*110540) < 100:
        raise HTTPException(status_code=422, detail="Điểm đi và điểm đến quá gần nhau; chọn hai địa điểm khác nhau.")
    try:
        base_url = os.getenv("ROUTING_API_BASE_URL") or None
        raw_routes = fetch_route_alternatives(body.origin.model_dump(), body.destination.model_dump(),
                                               base_url=base_url,
                                               timeout_seconds=float(os.getenv("ROUTING_API_TIMEOUT_SECONDS", "8")))
    except (RouteServiceError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Không lấy được tuyến đường thực từ dịch vụ chỉ đường: {str(exc)[:260]}") from exc
    import time
    now = time.time()
    with LIVE_TELEMETRY_LOCK:
        telemetry = [dict(item) for item in LIVE_TELEMETRY.values() if now - float(item.get("observed_epoch", 0)) <= 180]
    scored = score_routes(raw_routes, telemetry, now_epoch=now)
    analysis_context: dict[str, Any] = {"available": False, "notice": "Chưa gắn kết quả phân tích video với điểm đi."}
    if body.analysis_job_id:
        with VIDEO_REVIEW_LOCK:
            job = VIDEO_REVIEW_JOBS.get(body.analysis_job_id)
            result = job.get("result") if job and job.get("status") == "succeeded" else None
        if result:
            source_camera_id = result.get("source_camera_id")
            if body.origin.camera_id and source_camera_id == body.origin.camera_id:
                forecast = result.get("trend_forecast", {})
                predictions = forecast.get("predictions", []) if isinstance(forecast, dict) else []
                analysis_context = {"available": True, "source_camera_id": source_camera_id,
                    "source_state": (result.get("summary") or {}).get("current_state"),
                    "source_ci": (result.get("summary") or {}).get("current_ci"),
                    "trend_forecast": forecast, "source_is_origin": True}
                ten = next((p for p in predictions if p.get("horizon_minutes") == 10), None)
                if ten and float(ten.get("ci", 0)) >= 0.55:
                    analysis_context["risk_message"] = "Video ở điểm đi có tín hiệu nguy cơ ùn ứ ở mốc +10 phút theo baseline xu hướng; đây chưa phải dự báo đã huấn luyện."
                elif not forecast.get("available"):
                    analysis_context["risk_message"] = "Video cho biết trạng thái hiện tại theo luật hình ảnh; chưa đủ điều kiện ngoại suy +5/+10 phút."
            else:
                analysis_context = {"available": False, "notice": "Clip không thuộc camera đã chọn làm điểm đi nên không dùng để suy luận nguy cơ cho tuyến này."}
        else:
            analysis_context = {"available": False, "notice": "Không tìm thấy kết quả video hoàn tất hoặc tác vụ đã hết hạn."}
    scored["origin"] = body.origin.model_dump()
    scored["destination"] = body.destination.model_dump()
    scored["source_analysis"] = analysis_context
    scored["route_provider"] = "OSRM road network"
    scored["routing_mode"] = "live_telemetry_scored" if scored.get("traffic_aware") else "base_network_duration_only"
    scored["live_telemetry_points"] = len(telemetry)
    return scored


@app.post("/api/traffic/telemetry")
def ingest_live_traffic_telemetry(body: LiveTelemetryRequest) -> dict[str, Any]:
    """Collector contract for a protected RTSP process; never treat fixture camera metrics as live telemetry."""
    require_write_access()
    _assert_camera_exists(body.camera_id)
    camera = next((row for row in CAMERAS if row.get("id") == body.camera_id), None)
    if not camera:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")
    if not bool(camera.get("location_verified")):
        raise HTTPException(status_code=409, detail="Tọa độ camera chưa được xác minh. Cập nhật data/cameras.json trước khi gán telemetry vào bản đồ.")
    if camera.get("lat") is None or camera.get("lng") is None:
        raise HTTPException(status_code=422, detail="Camera chưa có tọa độ hợp lệ")
    now_dt = datetime.now(timezone.utc)
    observed = body.observed_at.astimezone(timezone.utc)
    age = (now_dt - observed).total_seconds()
    if age < -15 or age > 180:
        raise HTTPException(status_code=422, detail="Telemetry phải mới trong 180 giây; timestamp tương lai quá 15 giây bị từ chối.")
    record = {"camera_id": body.camera_id, "source": "rtsp", "observed_at": observed.isoformat(),
              "observed_epoch": observed.timestamp(), "lat": float(camera["lat"]), "lng": float(camera["lng"]),
              "ci": float(body.ci), "speed_kmh": body.speed_kmh, "flow_pcu": body.flow_pcu}
    with LIVE_TELEMETRY_LOCK:
        LIVE_TELEMETRY[body.camera_id] = record
        history = LIVE_TELEMETRY_HISTORY.setdefault(body.camera_id, [])
        if not history or record["observed_epoch"] > history[-1].get("observed_epoch", 0):
            history.append(dict(record))
        cutoff = now_dt.timestamp() - LIVE_TELEMETRY_HISTORY_MAX_AGE_SECONDS
        history[:] = [item for item in history if float(item.get("observed_epoch", 0)) >= cutoff][-LIVE_TELEMETRY_HISTORY_MAX_SAMPLES:]
        while len(LIVE_TELEMETRY) > MAX_LIVE_TELEMETRY_CAMERAS:
            oldest = min(LIVE_TELEMETRY, key=lambda key: LIVE_TELEMETRY[key].get("observed_epoch", 0))
            LIVE_TELEMETRY.pop(oldest, None)
            LIVE_TELEMETRY_HISTORY.pop(oldest, None)
    return {"ok": True, "camera_id": body.camera_id, "source": "rtsp", "observed_at": observed.isoformat(),
            "age_seconds": round(max(0, age), 1), "stored": True,
            "notice": "Đây là điểm tiếp nhận từ collector RTSP đã xác thực; endpoint không tự mở luồng RTSP.",
            "persistent": False}


@app.get("/api/traffic/segment-forecasts")
def segment_traffic_forecasts() -> dict[str, Any]:
    """Return a clearly-labeled linear baseline only when real, fresh history is sufficient."""
    now = datetime.now(timezone.utc).timestamp()
    live = _recent_live_telemetry()
    with LIVE_TELEMETRY_LOCK:
        history_snapshot = {key: [dict(row) for row in rows] for key, rows in LIVE_TELEMETRY_HISTORY.items()}

    items = []
    for camera in CAMERAS:
        # Forecasts are strictly limited to camera coordinates inside the Hanoi boundary.
        if not is_hanoi_coordinate(camera.get("lat"), camera.get("lng")):
            continue
        camera_id = str(camera.get("id", ""))
        live_record = live.get(camera_id)
        history = history_snapshot.get(camera_id, [])
        result = forecast_from_history(history, now_epoch=now)
        if not live_record:
            result = {
                **result,
                "available": False,
                "predictions": [],
                "reason": "no_fresh_telemetry",
                "message": "Chưa có telemetry thật mới trong 180 giây từ collector được phép.",
            }
        items.append({
            "camera_id": camera_id,
            "name": str(camera.get("name", camera_id)),
            "area": str(camera.get("area", "")),
            "lat": camera.get("lat"),
            "lng": camera.get("lng"),
            **result,
        })

    return {
        "ok": True,
        "scope": "H\u00e0 N\u1ed9i",
        "boundary_source": "data/geo/hanoi_boundary.geojson",
        "mode": "baseline_reference",
        "method": "linear_trend_baseline",
        "validated": False,
        "persistent": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "authorized_rtsp_collector",
        "message": "Dự báo baseline chỉ xuất hiện sau khi có đủ telemetry thật; chưa phải mô hình AI đã kiểm định.",
        "items": items,
    }

@app.get("/api/traffic/telemetry/status")
def live_traffic_telemetry_status() -> dict[str, Any]:
    import time
    now = time.time()
    with LIVE_TELEMETRY_LOCK:
        rows = list(LIVE_TELEMETRY.values())
    fresh = [item for item in rows if 0 <= now - float(item.get("observed_epoch", 0)) <= 180]
    return {"source": "rtsp_collector_contract", "fresh_camera_count": len(fresh),
            "max_age_seconds": 180, "persistent": False,
            "note": "Không có collector RTSP chạy thật mặc định; các fixture không được tính là telemetry live."}


@app.post("/api/analytics/ingest-frame")
def ingest_tracked_frame(body: AnalyticsFrameRequest) -> dict[str, Any]:
    """Ingest already-tracked detections from a future detector/ByteTrack worker.

    Coordinates and counting-line endpoints are normalized to [0, 1]. This API is
    a tracker-to-analytics boundary, not a camera capture or detector endpoint.
    """
    require_write_access()
    _assert_camera_exists(body.camera_id)
    try:
        with ANALYTICS_ENGINES_LOCK:
            engine = ANALYTICS_ENGINES.setdefault(body.camera_id, TrafficAnalyticsEngine())
            result = engine.process_frame(
                [item.model_dump() for item in body.tracks],
                timestamp_epoch=body.timestamp.timestamp(),
                counting_lines=[item.model_dump() for item in body.counting_lines],
                detector_confidence=body.detector_confidence,
            )
    except AnalyticsInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ok": True,
        "camera_id": body.camera_id,
        "timestamp": body.timestamp.isoformat(),
        "analytics": result,
        "notice": "Dữ liệu đầu vào là observations đã theo vết; endpoint này không tự chạy detector/RTSP. Các cảnh báo heuristic cần người vận hành xác minh.",
    }


@app.get("/api/analytics/cameras/{camera_id}/events")
def camera_analytics_events(camera_id: str, limit: int = 50) -> dict[str, Any]:
    """Read recent in-memory candidate events for a camera; not a persistent audit log."""
    _assert_camera_exists(camera_id)
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit phải nằm trong khoảng 1–200")
    with ANALYTICS_ENGINES_LOCK:
        engine = ANALYTICS_ENGINES.get(camera_id)
        events = engine.recent_events(limit) if engine else []
    return {"camera_id": camera_id, "items": events, "total": len(events), "persistent": False}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return {
        "cpu_latency_target_ms": 30,
        "forecast_latency_target_ms": 1,
        "ram_target_mb": 600,
        "fps_target": 8,
        "history": [
            {"day": (datetime.now().date() - timedelta(days=i)).strftime("%d/%m"), "availability": round(98.7 + (i % 4) * 0.25, 1), "alerts": max(4, 13 - i)}
            for i in reversed(range(7))
        ],
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
    logger.exception("Unhandled request error method=%s path=%s", request.method, request.url.path)
    # Keep internals and local file paths out of client-facing responses.
    return JSONResponse(status_code=500, content={"detail": "Lỗi xử lý yêu cầu. Kiểm tra log backend để biết chi tiết."})
