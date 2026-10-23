from __future__ import annotations

"""Provider adapter boundary for forecast models.

The default provider delegates to the current local ONNX runtime. A future model can
be exposed as a small HTTPS JSON API and selected with FORECAST_PROVIDER=remote_http.
Secrets are read from environment/.env only and are never included in status output.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CONTRACT_VERSION = "hanoi-traffic-forecast/2.0"
SUPPORTED_PROVIDERS = {"local_onnx", "remote_http"}


class ProviderConfigurationError(RuntimeError):
    """Raised when provider configuration or a remote provider response is invalid."""


def load_dotenv(path: Path) -> None:
    """Load a simple local .env without overriding real process environment variables."""
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key.replace("_", "").isalnum() and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # A missing or unreadable local override should not prevent a demo boot.
        return


def _allowed_remote_url(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    if parsed.scheme == "https" and parsed.hostname and parsed.username is None and parsed.password is None:
        return True
    # Permit plain HTTP only for local services during development.
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.username is None and parsed.password is None


def _label_for_ci(ci: float, thresholds: dict[str, Any]) -> tuple[str, str]:
    busy = float(thresholds.get("busy", 0.30))
    heavy = float(thresholds.get("heavy", 0.55))
    gridlock = float(thresholds.get("gridlock", 0.78))
    if ci < busy:
        return "Thông thoáng", "free"
    if ci < heavy:
        return "Đông đúc", "busy"
    if ci < gridlock:
        return "Ùn ứ cục bộ", "heavy"
    return "Tắc nghẽn nghiêm trọng", "gridlock"


class ForecastProviderManager:
    """Routes forecast requests to local ONNX inference or a configured HTTP API."""

    def __init__(self, runtime: Any, manifest_path: Path):
        self.runtime = runtime
        self.manifest_path = manifest_path

    def _manifest(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _provider_name(self) -> str:
        return os.getenv("FORECAST_PROVIDER", "local_onnx").strip().lower() or "local_onnx"

    def status(self) -> dict[str, Any]:
        name = self._provider_name()
        if name == "local_onnx":
            runtime_status = self.runtime.status()
            ready = bool(runtime_status.get("forecaster", {}).get("ready"))
            return {
                "provider": name,
                "display_name": "Mô hình cục bộ",
                "configured": True,
                "ready": ready,
                "mode": runtime_status.get("mode", "demo"),
                "endpoint_host": None,
                "timeout_seconds": None,
                "secret_configured": False,
                "message": "ONNX cục bộ đã sẵn sàng." if ready else "Chưa có forecaster ONNX hợp lệ; hệ thống dùng dữ liệu minh họa.",
            }
        if name == "remote_http":
            url = os.getenv("FORECAST_API_URL", "").strip()
            timeout = self._timeout_seconds()
            valid_url = bool(url and _allowed_remote_url(url))
            public_mode = os.getenv("PUBLIC_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
            public_enabled = os.getenv("FORECAST_API_PUBLIC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
            public_blocked = public_mode and not public_enabled
            parsed = urlparse(url) if url else None
            return {
                "provider": name,
                "display_name": "Dịch vụ dự báo bên ngoài",
                "configured": valid_url,
                "ready": valid_url and not public_blocked,
                "public_blocked": public_blocked,
                "mode": "remote_api" if valid_url and not public_blocked else "blocked" if public_blocked else "not_configured",
                "endpoint_host": parsed.hostname if parsed and valid_url and not public_blocked else None,
                "timeout_seconds": timeout,
                "secret_configured": bool(os.getenv("FORECAST_API_KEY", "").strip()),
                "message": "Dịch vụ API bị khóa trong public demo; chỉ bật khi đã thiết lập xác thực, hạn mức và giám sát chi phí." if public_blocked else "API dự báo đã được cấu hình; kết nối sẽ được kiểm tra khi gửi yêu cầu." if valid_url else "Cần cấu hình FORECAST_API_URL bằng HTTPS (hoặc localhost khi phát triển).",
            }
        return {
            "provider": name,
            "display_name": "Không rõ nhà cung cấp",
            "configured": False,
            "ready": False,
            "mode": "configuration_error",
            "endpoint_host": None,
            "timeout_seconds": None,
            "secret_configured": False,
            "message": "FORECAST_PROVIDER chỉ nhận local_onnx hoặc remote_http.",
        }

    @staticmethod
    def _timeout_seconds() -> float:
        try:
            value = float(os.getenv("FORECAST_API_TIMEOUT_SECONDS", "8"))
        except ValueError:
            value = 8.0
        return max(1.0, min(value, 60.0))

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        name = self._provider_name()
        if name == "local_onnx":
            result = self.runtime.predict(features)
            result["provider"] = name
            return result
        if name != "remote_http":
            return {"mode": "error", "provider": name, "error": "FORECAST_PROVIDER không hợp lệ.", "predictions": []}
        try:
            return self._predict_remote(features)
        except ProviderConfigurationError as exc:
            return {"mode": "error", "provider": name, "error": str(exc), "predictions": []}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # Invalid remote payloads/configuration must produce a controlled error, not a 500.
            return {"mode": "error", "provider": name, "error": f"Dữ liệu dự báo không hợp lệ ({type(exc).__name__}).", "predictions": []}

    def _predict_remote(self, features: dict[str, float]) -> dict[str, Any]:
        public_mode = os.getenv("PUBLIC_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
        public_enabled = os.getenv("FORECAST_API_PUBLIC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        if public_mode and not public_enabled:
            raise ProviderConfigurationError("API dự báo bên ngoài bị khóa ở chế độ public demo; chỉ mở sau khi cấu hình xác thực, giới hạn yêu cầu và giám sát chi phí.")
        url = os.getenv("FORECAST_API_URL", "").strip()
        if not url or not _allowed_remote_url(url):
            raise ProviderConfigurationError("Cấu hình API chưa hợp lệ: đặt FORECAST_API_URL bằng HTTPS; HTTP chỉ được phép với localhost.")
        manifest = self._manifest()
        schema = manifest.get("feature_schema", {})
        keys = schema.get("ordered_input_keys", [])
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
            raise ProviderConfigurationError("Manifest thiếu feature_schema.ordered_input_keys hợp lệ.")
        horizons = manifest.get("forecaster_config", {}).get("horizons_minutes", [5, 10, 15])
        if horizons != [5, 10, 15]:
            raise ProviderConfigurationError("Provider contract yêu cầu đúng thứ tự ba chân trời [5, 10, 15] phút.")
        clean_features: dict[str, float] = {}
        try:
            for key, value in features.items():
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Giá trị {key} không hữu hạn")
                clean_features[str(key)] = number
            missing = [key for key in keys if key not in clean_features]
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "…" if len(missing) > 5 else ""
                raise ProviderConfigurationError(f"Thiếu đặc trưng đo đạc bắt buộc cho API: {preview}{suffix}. Không tự điền fallback vào dự báo từ xa.")
            ordered_vector = [clean_features[key] for key in keys]
        except (ValueError, TypeError, OverflowError) as exc:
            raise ProviderConfigurationError(f"Đặc trưng đầu vào không hợp lệ: {type(exc).__name__}.") from exc
        request_payload = {
            "contract_version": CONTRACT_VERSION,
            "domain_profile": manifest.get("domain_profile", "unknown"),
            "manifest_version": manifest.get("artifact_version", "unknown"),
            "horizons_minutes": [int(h) for h in horizons],
            "ordered_input_keys": keys,
            "feature_vector": ordered_vector,
            "features": clean_features,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "HanoiTrafficPlatform/6.0"}
        api_key = os.getenv("FORECAST_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(url, data=json.dumps(request_payload, allow_nan=False).encode("utf-8"), headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(req, timeout=self._timeout_seconds()) as response:
                raw = response.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise ProviderConfigurationError("API trả body lớn hơn giới hạn 1 MB.")
                status_code = int(response.status)
        except HTTPError as exc:
            # Do not propagate response body or request headers, which could contain secrets.
            raise ProviderConfigurationError(f"API dự báo trả HTTP {exc.code}.") from exc
        except URLError as exc:
            raise ProviderConfigurationError(f"Không kết nối được API dự báo ({exc.reason.__class__.__name__}).") from exc
        except TimeoutError as exc:
            raise ProviderConfigurationError("API dự báo vượt thời gian chờ.") from exc
        if status_code < 200 or status_code >= 300:
            raise ProviderConfigurationError(f"API dự báo trả HTTP {status_code}.")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderConfigurationError("API dự báo không trả JSON hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise ProviderConfigurationError("API dự báo phải trả một JSON object.")
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(horizons):
            raise ProviderConfigurationError("Response API cần predictions gồm đúng ba phần tử (5, 10, 15 phút).")
        thresholds = manifest.get("forecaster_config", {}).get("thresholds", {"busy": 0.30, "heavy": 0.55, "gridlock": 0.78})
        normalized: list[dict[str, Any]] = []
        for expected_horizon, item in zip(horizons, predictions):
            try:
                actual_horizon = int(item.get("horizon_minutes", -1)) if isinstance(item, dict) else -1
            except (TypeError, ValueError, OverflowError):
                actual_horizon = -1
            if not isinstance(item, dict) or actual_horizon != int(expected_horizon):
                raise ProviderConfigurationError("Thứ tự horizon API không khớp manifest (cần 5, 10, 15 phút).")
            try:
                ci = float(item["ci"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ProviderConfigurationError("Mỗi prediction phải có trường số ci.") from exc
            if not math.isfinite(ci) or not 0.0 <= ci <= 1.0:
                raise ProviderConfigurationError("Chỉ số ci phải hữu hạn và nằm trong [0, 1].")
            level, level_key = _label_for_ci(ci, thresholds)
            normalized.append({"horizon_minutes": int(expected_horizon), "ci": round(ci, 4), "level": level, "level_key": level_key})
        return {
            "mode": "remote_api",
            "provider": "remote_http",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "predictions": normalized,
            "features_used": {key: clean_features.get(key, float(schema.get("default_fallback_values", {}).get(key, 0.0))) for key in keys},
        }
