from __future__ import annotations

import json
from pathlib import Path

from traffic_brain.providers import ForecastProviderManager


class _Runtime:
    def status(self):
        return {"mode": "demo", "forecaster": {"ready": False}}

    def predict(self, features):
        return {"mode": "demo", "predictions": [], "features_used": features}


class _FakeResponse:
    status = 200

    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit=-1):
        return self.body[:limit]


def _manifest(path: Path):
    path.write_text(json.dumps({
        "artifact_version": "2.0.0-schema-only",
        "domain_profile": "SEA_Urban_Unstructured_Hanoi",
        "feature_schema": {
            "ordered_input_keys": ["speed.mean_kmh", "flow.car_count"],
            "default_fallback_values": {"speed.mean_kmh": 42.0, "flow.car_count": 0.0},
        },
        "forecaster_config": {"horizons_minutes": [5, 10, 15], "thresholds": {"busy": .30, "heavy": .55, "gridlock": .78}},
    }), encoding="utf-8")


def test_local_provider_delegates(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manager = ForecastProviderManager(_Runtime(), manifest_path)
    monkeypatch.setenv("FORECAST_PROVIDER", "local_onnx")
    result = manager.predict({"speed.mean_kmh": 18.0})
    assert result["provider"] == "local_onnx"
    assert result["mode"] == "demo"


def test_remote_provider_sends_contract_and_normalizes_output(monkeypatch, tmp_path):
    import traffic_brain.providers as providers

    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    calls = {}

    def fake_urlopen(request, timeout):
        calls["request"] = request
        calls["timeout"] = timeout
        return _FakeResponse({"contract_version": "hanoi-traffic-forecast/2.0", "predictions": [
            {"horizon_minutes": 5, "ci": 0.21},
            {"horizon_minutes": 10, "ci": 0.46},
            {"horizon_minutes": 15, "ci": 0.81},
        ]})

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    monkeypatch.setenv("FORECAST_PROVIDER", "remote_http")
    monkeypatch.setenv("FORECAST_API_URL", "https://forecast.example.test/predict")
    monkeypatch.setenv("FORECAST_API_KEY", "top-secret-test-value")
    monkeypatch.setenv("FORECAST_API_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")

    manager = ForecastProviderManager(_Runtime(), manifest_path)
    result = manager.predict({"speed.mean_kmh": 17.5, "flow.car_count": 32})

    assert result["mode"] == "remote_api"
    assert [p["horizon_minutes"] for p in result["predictions"]] == [5, 10, 15]
    assert result["predictions"][0]["level_key"] == "free"
    assert result["predictions"][-1]["level_key"] == "gridlock"
    assert calls["timeout"] == 4.0
    assert calls["request"].get_header("Authorization") == "Bearer top-secret-test-value"
    sent = json.loads(calls["request"].data.decode("utf-8"))
    assert sent["contract_version"] == "hanoi-traffic-forecast/2.0"
    assert sent["feature_vector"] == [17.5, 32.0]
    assert "top-secret-test-value" not in json.dumps(manager.status())


def test_remote_provider_rejects_wrong_horizons(monkeypatch, tmp_path):
    import traffic_brain.providers as providers

    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    monkeypatch.setattr(providers, "urlopen", lambda request, timeout: _FakeResponse({"predictions": [
        {"horizon_minutes": 5, "ci": .2}, {"horizon_minutes": 15, "ci": .4}, {"horizon_minutes": 10, "ci": .5}
    ]}))
    monkeypatch.setenv("FORECAST_PROVIDER", "remote_http")
    monkeypatch.setenv("FORECAST_API_URL", "https://forecast.example.test/predict")
    monkeypatch.setenv("FORECAST_API_KEY", "")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    manager = ForecastProviderManager(_Runtime(), manifest_path)
    result = manager.predict({"speed.mean_kmh": 20.0, "flow.car_count": 10.0})
    assert result["mode"] == "error"
    assert "horizon" in result["error"].lower()


def test_public_demo_blocks_remote_provider_by_default(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    monkeypatch.setenv("FORECAST_PROVIDER", "remote_http")
    monkeypatch.setenv("FORECAST_API_URL", "https://forecast.example.test/predict")
    monkeypatch.setenv("FORECAST_API_PUBLIC_ENABLED", "false")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    manager = ForecastProviderManager(_Runtime(), manifest_path)
    status = manager.status()
    result = manager.predict({})
    assert status["public_blocked"] is True
    assert result["mode"] == "error"
    assert "public demo" in result["error"].lower()


def test_remote_provider_requires_secure_url(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    monkeypatch.setenv("FORECAST_PROVIDER", "remote_http")
    monkeypatch.setenv("FORECAST_API_URL", "http://forecast.example.test/predict")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    manager = ForecastProviderManager(_Runtime(), manifest_path)
    assert manager.predict({})["mode"] == "error"


def test_remote_provider_rejects_missing_features_instead_of_inserting_fallbacks(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    monkeypatch.setenv("FORECAST_PROVIDER", "remote_http")
    monkeypatch.setenv("FORECAST_API_URL", "https://forecast.example.test/predict")
    monkeypatch.setenv("FORECAST_API_KEY", "")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    manager = ForecastProviderManager(_Runtime(), manifest_path)
    result = manager.predict({"flow.car_count": 12.0})
    assert result["mode"] == "error"
    assert "Thiếu đặc trưng đo đạc bắt buộc" in result["error"]
