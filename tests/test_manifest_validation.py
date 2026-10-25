from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def _valid_manifest():
    return copy.deepcopy(main.DEFAULT_MANIFEST)


def test_manifest_normalizes_canonical_relative_artifact_paths():
    payload = _valid_manifest()
    payload["detector_config"]["onnx_path"] = "detector.onnx"
    payload["forecaster_config"]["onnx_path"] = "forecaster.onnx"
    normalized = main._normalize_manifest_payload(payload)
    assert normalized["detector_config"]["onnx_path"] == "artifacts/current/detector.onnx"
    assert normalized["forecaster_config"]["onnx_path"] == "artifacts/current/forecaster.onnx"
    # Normalization must not mutate the caller's object.
    assert payload["detector_config"]["onnx_path"] == "detector.onnx"


@pytest.mark.parametrize("bad_path", ["../../secret.onnx", "/tmp/detector.onnx", "C:/temp/detector.onnx", "artifacts/current/../detector.onnx"])
def test_manifest_rejects_paths_outside_artifact_contract(bad_path):
    payload = _valid_manifest()
    payload["detector_config"]["onnx_path"] = bad_path
    with pytest.raises(ValueError):
        main._normalize_manifest_payload(payload)


def test_manifest_rejects_duplicate_feature_keys():
    payload = _valid_manifest()
    payload["feature_schema"]["ordered_input_keys"] = ["speed.mean_kmh", "speed.mean_kmh"]
    with pytest.raises(ValueError, match="không được lặp"):
        main._normalize_manifest_payload(payload)


def test_manifest_rejects_wrong_forecast_horizon_order():
    payload = _valid_manifest()
    payload["forecaster_config"]["horizons_minutes"] = [5, 15, 10]
    with pytest.raises(ValueError, match=r"\[5, 10, 15\]"):
        main._normalize_manifest_payload(payload)


def test_manifest_rejects_unordered_ci_thresholds():
    payload = _valid_manifest()
    payload["forecaster_config"]["thresholds"] = {"busy": 0.6, "heavy": 0.5, "gridlock": 0.9}
    with pytest.raises(ValueError, match="busy=0.30, heavy=gridlock=0.55"):
        main._normalize_manifest_payload(payload)


def test_forecast_endpoint_rejects_nonfinite_values():
    response = client.post("/api/forecast", content='{"features":{"speed.mean_kmh":NaN}', headers={"content-type": "application/json"})
    assert response.status_code == 422
