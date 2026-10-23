from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_aicc_status_is_available_without_model_weights():
    response = client.get('/api/aicc/status')
    assert response.status_code == 200
    payload = response.json()
    assert 'enabled' in payload
    assert 'missing_files' in payload
    assert payload['provider'] == 'CPUExecutionProvider' or payload.get('provider') is None


def test_aicc_video_listing_endpoint_is_available():
    response = client.get('/api/aicc/data-videos')
    assert response.status_code == 200
    payload = response.json()
    assert 'items' in payload
    assert isinstance(payload['items'], list)
