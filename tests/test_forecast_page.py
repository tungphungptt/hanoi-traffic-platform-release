"""Home page = forecast verification; old dashboard moved to /dashboard; cross-video library."""
from fastapi.testclient import TestClient

import app.main as main
from tests.test_minute_forecaster import episode_timeline

client = TestClient(main.app)


def test_home_is_forecast_verification_page():
    html = client.get('/').text
    for marker in ('id="videoFile"', 'id="player"', 'id="ciChart"', 'id="horizonList"', 'id="blindMode"',
                   'id="metricsTable"', '/static/forecast.js', 'href="/dashboard"'):
        assert marker in html
    assert '+5/+10/+15' in html or '5, 10 và 15 phút' in html
    assert client.get('/static/forecast.js').status_code == 200
    assert client.get('/static/forecast.css').status_code == 200


def test_previous_dashboard_still_served():
    html = client.get('/dashboard').text
    assert '<section class="page active" id="page-video">' in html


def test_status_exposes_library_and_long_local_limit(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', False)
    status = client.get('/api/aicc/status').json()
    assert 'forecast_library_enabled' in status
    assert status['upload_max_duration_seconds'] >= 30 * 60


def test_library_never_returns_the_same_video(monkeypatch):
    monkeypatch.setattr(main, 'FORECAST_LIBRARY_ENABLED', True)
    timeline = episode_timeline(5, duration=12 * 60)
    main._save_forecast_library('a' * 32, 'one.mp4', timeline)
    main._save_forecast_library('b' * 32, 'two.mp4', timeline)
    assert len(main._load_forecast_library('a' * 32)) == 1
    assert len(main._load_forecast_library('c' * 32)) == 2
    monkeypatch.setattr(main, 'FORECAST_LIBRARY_ENABLED', False)
    assert main._load_forecast_library(None) == []


def test_worker_passes_library_and_saves_series(monkeypatch, tmp_path):
    import cv2
    import numpy as np
    monkeypatch.setattr(main, 'FORECAST_LIBRARY_ENABLED', True)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_ROOT', tmp_path / 'reviews')
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {'f' * 32: {'status': 'queued'}})
    monkeypatch.setattr(main.aicc_runtime, 'status', lambda: {'missing_files': []})
    monkeypatch.setattr(main.aicc_runtime, 'load', lambda: {})
    monkeypatch.setattr(main.hanoi_video_forecaster, 'load', lambda: {})
    clip = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 5, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    seen = {}

    def analyze(path, bundle, output_dir, **kwargs):
        seen.update(kwargs)
        return {'frames': [], 'timeline': episode_timeline(6, duration=12 * 60)}

    monkeypatch.setattr(main, 'analyze_video', analyze)
    main._run_video_review_job('f' * 32, clip, clip.name, None, True, False)
    assert main.VIDEO_REVIEW_JOBS['f' * 32]['status'] == 'succeeded'
    assert seen['forecast_library'] == [] and seen['forecast_library_videos'] == 0
    assert len(list(main.FORECAST_LIBRARY_DIR.glob('*.json'))) == 1


def test_bundled_training_library_is_valid_and_loaded(monkeypatch):
    import json
    from traffic_brain import minute_forecaster as mf
    files = sorted(main.BUNDLED_FORECAST_LIBRARY_DIR.glob('*.json'))
    assert len(files) >= 2, 'the shipped forecaster must have a training library'
    for path in files:
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['video_key'] == path.stem
        series = mf.record_to_series(record)
        assert series is not None and series['t'].size >= 8
        assert int((series['ci'] == series['ci']).sum()) >= 8  # at least 2 observed minutes
    monkeypatch.setattr(main, 'BUNDLED_FORECAST_LIBRARY_ENABLED', True)
    assert len(main._load_forecast_library(None)) == len(files)
    assert len(main._load_forecast_library(files[0].stem)) == len(files) - 1
    # Records cut from a recording are skipped when that recording itself is uploaded.
    for path in files:
        record = json.loads(path.read_text(encoding='utf-8'))
        sources = record.get('source_video_keys') or []
        if sources:
            sharing = sum(1 for p in files if sources[0] in (json.loads(p.read_text(encoding='utf-8')).get('source_video_keys') or []))
            assert len(main._load_forecast_library(sources[0])) == len(files) - sharing
            break
    status = client.get('/api/aicc/status').json()
    assert status['forecast_library_videos'] >= len(files)
