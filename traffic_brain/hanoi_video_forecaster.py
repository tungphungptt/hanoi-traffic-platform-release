"""Inference runtime for the uploaded-video Hanoi multi-horizon LSTM."""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hanoi_predictor.onnx"
SCALER_PATH = MODEL_DIR / "hanoi_scaler.pkl"
MANIFEST_PATH = MODEL_DIR / "hanoi_manifest.json"
FEATURES = (
    "car_count",
    "moto_count",
    "total_vehicles",
    "avg_speed",
    "occupancy",
    "speed_cv",
    "brightness",
    "contrast",
    "saturation",
)
HORIZONS_SECONDS = (5, 10, 15)
SEQUENCE_LENGTH = 30


class HanoiVideoForecaster:
    """Load and run the trusted ONNX/scaler bundle for video sequences."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundle: dict[str, Any] | None = None
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._error: str | None = None

    def _artifact_signature(self) -> tuple[tuple[str, int, int], ...]:
        files = (MODEL_PATH, SCALER_PATH, MANIFEST_PATH)
        return tuple(
            (
                path.name,
                path.stat().st_mtime_ns if path.is_file() else -1,
                path.stat().st_size if path.is_file() else -1,
            )
            for path in files
        )

    def status(self) -> dict[str, Any]:
        """Report bundle presence and load state without loading a pickle."""
        missing = [
            path.relative_to(ROOT).as_posix()
            for path in (MODEL_PATH, SCALER_PATH, MANIFEST_PATH)
            if not path.is_file()
        ]
        signature = self._artifact_signature()
        loaded = self._bundle is not None and signature == self._signature
        return {
            "ready": not missing and loaded,
            "artifacts_present": not missing,
            "loaded": loaded,
            "missing_files": missing,
            "model_version": "v2.0-hanoi-forecast",
            "sequence_length": SEQUENCE_LENGTH,
            "features": list(FEATURES),
            "horizons_seconds": list(HORIZONS_SECONDS),
            "provider": "CPUExecutionProvider",
            "error": self._error,
        }

    def load(self, force: bool = False) -> dict[str, Any]:
        """Validate and load the manifest, ONNX model, and trusted scaler."""
        with self._lock:
            missing = [
                path.relative_to(ROOT).as_posix()
                for path in (MODEL_PATH, SCALER_PATH, MANIFEST_PATH)
                if not path.is_file()
            ]
            if missing:
                self._bundle = None
                self._signature = None
                self._error = "Thiếu Hanoi forecast artifacts: " + ", ".join(missing)
                raise FileNotFoundError(self._error)

            signature = self._artifact_signature()
            if not force and self._bundle is not None and signature == self._signature:
                return self._bundle

            try:
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("Hanoi manifest phải là một JSON object")
                if manifest.get("model_format") != "onnx":
                    raise ValueError("Hanoi model_format phải là onnx")
                if manifest.get("sequence_length") != SEQUENCE_LENGTH:
                    raise ValueError(
                        f"sequence_length phải là {SEQUENCE_LENGTH}"
                    )
                if tuple(manifest.get("features", ())) != FEATURES:
                    raise ValueError("Thứ tự features trong Hanoi manifest không khớp")
                if tuple(manifest.get("horizons_seconds", ())) != HORIZONS_SECONDS:
                    raise ValueError("Hanoi horizons_seconds phải là [5, 10, 15]")

                import onnxruntime as ort
                from sklearn.preprocessing import MinMaxScaler

                # The scaler is a trusted first-party artifact shipped with this app.
                with SCALER_PATH.open("rb") as stream:
                    scaler = pickle.load(stream)
                if not isinstance(scaler, MinMaxScaler):
                    raise TypeError("hanoi_scaler.pkl phải là MinMaxScaler")
                if getattr(scaler, "n_features_in_", None) != len(FEATURES):
                    raise ValueError("Hanoi scaler phải được fit trên 9 features")

                session = ort.InferenceSession(
                    str(MODEL_PATH),
                    providers=["CPUExecutionProvider"],
                )
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if len(inputs) != 1 or len(outputs) != 1:
                    raise ValueError("Hanoi ONNX cần đúng một input và một output")
                input_shape = inputs[0].shape
                if len(input_shape) != 3:
                    raise ValueError(f"Hanoi ONNX input phải là rank 3: {input_shape}")
                for actual, expected, field in zip(
                    input_shape[1:],
                    (SEQUENCE_LENGTH, len(FEATURES)),
                    ("sequence_length", "feature_count"),
                ):
                    if isinstance(actual, int) and actual != expected:
                        raise ValueError(
                            f"Hanoi ONNX {field} phải là {expected}; got {actual}"
                        )
                output_shape = outputs[0].shape
                if len(output_shape) != 2 or (
                    isinstance(output_shape[1], int)
                    and output_shape[1] != len(HORIZONS_SECONDS)
                ):
                    raise ValueError(
                        "Hanoi ONNX output phải có shape (batch, 3); "
                        f"got {output_shape}"
                    )

                self._bundle = {
                    "manifest": manifest,
                    "scaler": scaler,
                    "session": session,
                    "input_name": inputs[0].name,
                    "output_name": outputs[0].name,
                }
                self._signature = signature
                self._error = None
                return self._bundle
            except Exception as exc:
                self._bundle = None
                self._signature = None
                self._error = f"Không nạp được Hanoi LSTM: {exc}"
                raise RuntimeError(self._error) from exc

    def predict(self, sequence: np.ndarray) -> dict[str, Any]:
        """Predict congestion at +5/+10/+15 seconds for 30 feature rows."""
        values = np.asarray(sequence, dtype=np.float32)
        expected_shape = (SEQUENCE_LENGTH, len(FEATURES))
        if values.shape != expected_shape:
            raise ValueError(
                f"Hanoi LSTM input phải có shape {expected_shape}; got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Hanoi LSTM input chứa giá trị không hữu hạn")

        bundle = self.load()
        scaled = bundle["scaler"].transform(values).astype(np.float32)
        tensor = np.ascontiguousarray(scaled[np.newaxis, ...])
        raw_output = bundle["session"].run(
            [bundle["output_name"]],
            {bundle["input_name"]: tensor},
        )[0]
        logits = np.asarray(raw_output, dtype=np.float32)
        if logits.shape != (1, len(HORIZONS_SECONDS)) or not np.isfinite(logits).all():
            raise ValueError(
                "Hanoi LSTM trả output không hợp lệ; "
                f"expected (1, 3), got {logits.shape}"
            )
        logits = logits[0]
        probabilities = np.empty_like(logits)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)

        predictions = []
        for horizon, probability in zip(HORIZONS_SECONDS, probabilities):
            congestion_probability = float(probability)
            predictions.append(
                {
                    "horizon_seconds": horizon,
                    "congestion_probability": round(congestion_probability, 4),
                    "predicted_class": (
                        "Tac_duong" if congestion_probability >= 0.5 else "Thong_thoang"
                    ),
                }
            )

        return {
            "available": True,
            "mode": "experimental_lstm",
            "model_version": bundle["manifest"]["version"],
            "predictions": predictions,
            "probability_calibrated": False,
            "message": (
                "Dự báo thử nghiệm +5/+10/+15 giây từ chuỗi video 1 Hz; "
                "điểm xác suất chưa được hiệu chuẩn."
            ),
        }


hanoi_video_forecaster = HanoiVideoForecaster()
