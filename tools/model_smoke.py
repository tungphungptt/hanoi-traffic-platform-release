#!/usr/bin/env python3
"""Load the complete trusted AICC bundle and execute bounded CPU inference checks.

This checks artifact compatibility and ONNX execution only. It is not an accuracy benchmark.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fixed_dimension(actual: Any, expected: int, name: str) -> int:
    """Require a known fixed dimension to match; resolve symbolic dimensions safely."""
    if isinstance(actual, int) and not isinstance(actual, bool):
        if actual != expected:
            raise ValueError(f"{name} dimension {actual} does not match expected {expected}")
    elif actual is not None and str(actual).isdigit():
        if int(actual) != expected:
            raise ValueError(f"{name} dimension {actual} does not match expected {expected}")
    return expected


def _validate_finite_outputs(outputs: Sequence[Any], label: str) -> list[np.ndarray]:
    if not outputs:
        raise ValueError(f"{label} returned no outputs")
    arrays: list[np.ndarray] = []
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        if array.size == 0:
            raise ValueError(f"{label} output {index} is empty")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"{label} output {index} contains NaN/Inf")
        arrays.append(array)
    return arrays


def run_smoke() -> dict[str, Any]:
    from app.main import aicc_runtime

    started = time.perf_counter()
    bundle = aicc_runtime.load(force=True)
    manifest = bundle["manifest"]
    detector = bundle["detector"]
    predictor = bundle["predictor"]
    scaler = bundle["scaler"]

    if "CPUExecutionProvider" not in detector.get_providers():
        raise RuntimeError("Detector is not running with CPUExecutionProvider")
    if "CPUExecutionProvider" not in predictor.get_providers():
        raise RuntimeError("Predictor is not running with CPUExecutionProvider")

    detector_meta = detector.get_inputs()[0]
    detector_shape = detector_meta.shape
    if len(detector_shape) != 4:
        raise ValueError(f"Detector input must be NCHW rank 4, got {detector_shape}")
    _fixed_dimension(detector_shape[0], 1, "detector batch")
    channels = _fixed_dimension(detector_shape[1], 3, "detector channels")
    height = _fixed_dimension(detector_shape[2], 640, "detector height")
    width = _fixed_dimension(detector_shape[3], 640, "detector width")
    detector_tensor = np.zeros((1, channels, height, width), dtype=np.float32)
    detector_started = time.perf_counter()
    detector_outputs = _validate_finite_outputs(
        detector.run(None, {detector_meta.name: detector_tensor}), "detector"
    )
    detector_ms = (time.perf_counter() - detector_started) * 1000.0

    sequence_length = manifest.get("sequence_length")
    features = manifest.get("features")
    classes = manifest.get("classes")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("AICC manifest sequence_length must be a positive integer")
    if not isinstance(features, list) or not features or any(not isinstance(x, str) for x in features):
        raise ValueError("AICC manifest features must be a non-empty string list")
    if not isinstance(classes, list) or not classes or any(not isinstance(x, str) for x in classes):
        raise ValueError("AICC manifest classes must be a non-empty string list")
    scaler_features = getattr(scaler, "n_features_in_", len(features))
    if int(scaler_features) != len(features):
        raise ValueError(
            f"Scaler feature count {scaler_features} disagrees with manifest features {len(features)}"
        )

    predictor_meta = predictor.get_inputs()[0]
    predictor_shape = predictor_meta.shape
    if len(predictor_shape) != 3:
        raise ValueError(f"Predictor input must be [batch, sequence, features], got {predictor_shape}")
    _fixed_dimension(predictor_shape[0], 1, "predictor batch")
    _fixed_dimension(predictor_shape[1], sequence_length, "predictor sequence length")
    _fixed_dimension(predictor_shape[2], len(features), "predictor feature count")
    raw = np.zeros((sequence_length, len(features)), dtype=np.float32)
    scaled = np.asarray(scaler.transform(raw), dtype=np.float32)
    if scaled.shape != raw.shape or not np.isfinite(scaled).all():
        raise ValueError("Scaler output has invalid shape or contains NaN/Inf")
    predictor_tensor = np.ascontiguousarray(scaled[np.newaxis, ...], dtype=np.float32)
    predictor_started = time.perf_counter()
    predictor_outputs = _validate_finite_outputs(
        predictor.run(None, {predictor_meta.name: predictor_tensor}), "predictor"
    )
    predictor_ms = (time.perf_counter() - predictor_started) * 1000.0
    logits = predictor_outputs[0]
    if logits.shape[0] != 1 or int(np.prod(logits.shape[1:])) != len(classes):
        raise ValueError(
            f"Predictor output shape {logits.shape} does not match manifest classes count {len(classes)}"
        )

    return {
        "status": "pass",
        "model_version": manifest.get("version", "unknown"),
        "provider": "CPUExecutionProvider",
        "detector_input_shape": list(detector_tensor.shape),
        "detector_output_shapes": [list(x.shape) for x in detector_outputs],
        "predictor_input_shape": list(predictor_tensor.shape),
        "predictor_output_shape": list(logits.shape),
        "features": features,
        "classes": classes,
        "detector_tensor_inference_ms": round(detector_ms, 2),
        "predictor_tensor_inference_ms": round(predictor_ms, 2),
        "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "accuracy_claim": False,
        "note": "Synthetic zero-tensor smoke only; does not validate video quality, training provenance, Hanoi accuracy, or forecasting horizons.",
    }


def main() -> int:
    try:
        report = run_smoke()
    except Exception as exc:  # diagnostic should fail closed but avoid printing secrets
        print(f"MODEL_SMOKE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("MODEL_SMOKE_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
