from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_overview_has_camera_specific_video_player_and_illustrative_street_fallback():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'class="traffic-monitor-grid"' in html
    assert 'id="trafficCameraVideo"' in html
    assert 'id="trafficCameraEmpty"' in html
    assert 'src="/static/street-traffic-demo.jpg"' in html
    assert "không phải camera trực tiếp" in html.lower()


def test_model_artifacts_are_in_collapsed_technical_area_not_overview_cards():
    html = client.get("/").text
    assert '<details class="advanced-artifact-settings">' in html
    assert 'id="providerUiTitle"' in html
    assert 'id="detectorFileName"' in html
    overview = html.split('<section class="page" id="page-overview">', 1)[1].split('<section class="page" id="page-cameras">', 1)[0]
    assert 'id="detectorFileName"' not in overview
    assert 'id="forecasterFileName"' not in overview


def test_data_extension_bases_are_present_for_future_training_and_camera_feeds():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for relative in ("data/cameras.json", "data/camera_feeds/README.md", "data/observations/README.md", "data/labels/README.md", ".env.example", "docs/FORECAST_PROVIDER_CONTRACT.md"):
        assert (root / relative).is_file(), relative


def test_overview_uses_interactive_real_map_not_schematic_svg():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root / "static/index.html").read_text(encoding="utf-8")
    js = (root / "static/app.js").read_text(encoding="utf-8")
    assert 'id="leafletMap"' in html
    assert 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' in (root / "app/main.py").read_text(encoding="utf-8")
    assert 'data-map-layer="traffic"' in html
    assert 'mapHanoiFit' in html
    assert 'OpenStreetMap contributors' in js or 'OpenStreetMap contributors' in (root / "app/main.py").read_text(encoding="utf-8")
    assert 'L.map(' in js and 'pollCameraMapData' in js
    assert 'map-art' not in js


def test_scenario_first_landing_and_no_synthetic_dashboard_forecast():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root / "static/index.html").read_text(encoding="utf-8")
    js = (root / "static/app.js").read_text(encoding="utf-8")
    backend = (root / "app/main.py").read_text(encoding="utf-8")
    assert '<section class="page active" id="page-video">' in html
    assert "currentPage: 'video'" in js
    assert "Không tạo đường xu hướng từ số liệu mẫu" in js
    assert 'forecast_features = {' not in backend
    assert '"trend": [],' in backend


def test_video_upload_works_without_camera_selection_and_shows_experimental_hanoi_horizons():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    html = (root / "static/index.html").read_text(encoding="utf-8")
    js = (root / "static/app.js").read_text(encoding="utf-8")
    backend = (root / "app/main.py").read_text(encoding="utf-8")
    render = (root / "render.yaml").read_text(encoding="utf-8")
    assert "Đoạn A / điểm đi (không bắt buộc)" in html
    assert "LSTM thử nghiệm +5 / +10 / +15 phút" in html
    assert "if (originId) form.append('camera_id', originId)" in js
    assert "canAnalyzeUploadedScenarioVideo" in js
    assert "PUBLIC_VIDEO_REVIEW_ENABLED" in backend
    assert 'key: PUBLIC_VIDEO_REVIEW_ENABLED' in render


def test_alerts_are_not_fabricated_when_no_live_event_generator_exists():
    from fastapi.testclient import TestClient
    from app.main import app
    payload = TestClient(app).get('/api/alerts').json()
    assert payload['items'] == []
    assert payload['total'] == 0
    assert payload['source'] == 'not_connected'
