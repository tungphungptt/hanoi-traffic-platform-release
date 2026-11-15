import pytest
import app.main as main
from app.public_upload import UploadGate


@pytest.fixture(autouse=True)
def isolate_public_upload_quota(monkeypatch, tmp_path):
    # Global anti-abuse state belongs to a server instance, not to another unit test.
    monkeypatch.setattr(main, 'PUBLIC_UPLOAD_GATE', UploadGate())
    monkeypatch.setattr(main, 'VIDEO_REVIEW_FUTURES', {})
    # Cross-video forecast library must never read or write the developer's real data/.
    monkeypatch.setattr(main, 'FORECAST_LIBRARY_DIR', tmp_path / 'forecast_library')
