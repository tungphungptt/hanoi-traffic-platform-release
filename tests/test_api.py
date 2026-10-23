from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['ok'] is True


def test_dashboard_has_demo_or_production_mode():
    response = client.get('/api/dashboard')
    assert response.status_code == 200
    payload = response.json()
    assert payload['mode'] in ('demo', 'production', 'error')
    assert len(payload['cameras']) >= 1
    assert 'forecast' in payload


def test_forecast_accepts_features():
    response = client.post('/api/forecast', json={'features': {'speed.mean_kmh': 18.0, 'density.areal_occupancy': 0.45}})
    assert response.status_code == 200
    assert 'mode' in response.json()


def test_manifest_schema_exists():
    response = client.get('/api/models/manifest')
    assert response.status_code == 200
    assert 'ordered_input_keys' in response.json()['feature_schema']


def test_forecaster_refuses_to_reuse_one_output_for_three_horizons(monkeypatch):
    import numpy as np
    import app.main as main

    class InputMeta:
        name = "features"
        shape = [1, 8]

    class SingleOutputSession:
        def get_inputs(self):
            return [InputMeta()]

        def run(self, _outputs, _inputs):
            return [np.asarray([[0.52]], dtype=np.float32)]

    monkeypatch.setattr(main.runtime, "forecaster_session", SingleOutputSession())
    result = main.runtime.predict({})
    assert result["mode"] == "error"
    assert "trả 1 giá trị" in result["error"]
    assert result["predictions"] == []


def test_street_scene_placeholder_asset_is_served():
    response = client.get('/static/street-traffic-demo.jpg')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('image/jpeg')
    assert len(response.content) > 1000


def test_model_status_exposes_provider_without_secret(monkeypatch):
    import app.main as main
    monkeypatch.setenv('FORECAST_PROVIDER', 'remote_http')
    monkeypatch.setenv('FORECAST_API_URL', 'https://model.example.test/predict')
    monkeypatch.setenv('FORECAST_API_KEY', 'do-not-expose-this-secret')
    response = client.get('/api/models/status')
    assert response.status_code == 200
    payload = response.json()
    assert 'detector' in payload and 'forecaster' in payload
    assert payload['forecast_provider']['provider'] == 'remote_http'
    assert payload['forecast_provider']['secret_configured'] is True
    assert 'do-not-expose-this-secret' not in response.text
    monkeypatch.setenv('FORECAST_PROVIDER', 'local_onnx')


def test_observations_endpoint_builds_features_then_routes_to_forecast_provider(monkeypatch):
    import app.main as main
    monkeypatch.setenv('FORECAST_PROVIDER', 'local_onnx')
    observations = [
        {"track_id": "m1", "class_id": 0, "speed_kmh": 14.0, "active_now": True},
        {"track_id": "c1", "class_id": 1, "speed_kmh": 22.0, "active_now": True},
        {"track_id": "b1", "class_id": 2, "speed_kmh": 11.0, "active_now": True},
    ]
    response = client.post('/api/forecast/from-observations', json={
        "observations": observations,
        "previous_features": {"flow.pcu_equivalent": 3.0, "speed.mean_kmh": 18.0}
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['feature_quality']['ready_for_real_forecast'] is True
    assert payload['features']['flow.pcu_equivalent'] > 0
    assert payload['forecast']['provider'] == 'local_onnx'
    assert payload['mode'] in ('demo', 'production', 'error')


def test_observations_endpoint_refuses_to_forecast_with_missing_measured_features(monkeypatch):
    import app.main as main
    monkeypatch.setenv('FORECAST_PROVIDER', 'local_onnx')
    response = client.post('/api/forecast/from-observations', json={
        "observations": [{"track_id": "c1", "class_id": 1}],
        "previous_features": {}
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['mode'] == 'insufficient_features'
    assert payload['predictions'] == []
    assert payload['feature_quality']['ready_for_real_forecast'] is False


def test_map_config_exposes_real_hanoi_basemap_and_explicit_live_traffic_toggle(monkeypatch):
    import app.main as main
    monkeypatch.delenv("TRAFFIC_TILE_URL_TEMPLATE", raising=False)
    response = client.get("/api/map/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["tile_url_template"] == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    assert "OpenStreetMap" in payload["base_map_name"]
    assert payload["traffic_layer_enabled"] is False
    assert payload["camera_refresh_seconds"] >= 10


def test_map_config_accepts_configured_https_traffic_layer(monkeypatch):
    monkeypatch.setenv("TRAFFIC_TILE_URL_TEMPLATE", "https://traffic.example.test/tiles/{z}/{x}/{y}.png?key=browser-restricted")
    monkeypatch.setenv("TRAFFIC_TILE_ATTRIBUTION", "Traffic provider")
    response = client.get("/api/map/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["traffic_layer_enabled"] is True
    assert payload["traffic_tile_url_template"].startswith("https://traffic.example.test/")
    assert payload["traffic_attribution"] == "Traffic provider"
    monkeypatch.delenv("TRAFFIC_TILE_URL_TEMPLATE", raising=False)
    monkeypatch.delenv("TRAFFIC_TILE_ATTRIBUTION", raising=False)


def test_map_config_rejects_non_https_remote_traffic_tiles(monkeypatch):
    monkeypatch.setenv("TRAFFIC_TILE_URL_TEMPLATE", "http://traffic.example.test/{z}/{x}/{y}.png")
    response = client.get("/api/map/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["traffic_layer_enabled"] is False
    assert payload["traffic_tile_url_template"] == ""
    assert "HTTPS" in payload["configuration_warning"]
    monkeypatch.delenv("TRAFFIC_TILE_URL_TEMPLATE", raising=False)


def test_camera_map_points_include_real_geographic_coordinates_and_are_marked_fixture():
    response = client.get("/api/cameras")
    assert response.status_code == 200
    payload = response.json()
    assert payload["telemetry_mode"] == "not_connected"
    assert payload["demo"] is True
    assert payload["items"] and all(item["telemetry_live"] is False for item in payload["items"])
    assert all(item["ci"] is None and item["speed"] is None for item in payload["items"])
    assert payload["demo"] is True
    assert any(isinstance(row.get("lat"), (int, float)) and isinstance(row.get("lng"), (int, float)) for row in payload["items"])
