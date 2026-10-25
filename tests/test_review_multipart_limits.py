"""Only video review admits one spooled file and two small metadata fields."""
import asyncio
import tempfile

from fastapi.testclient import TestClient
import pytest
import starlette.formparsers

import app.main as main


@pytest.fixture(params=[False, True], ids=['local', 'public'])
def review_client(request, monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', request.param)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, 'AICC_VIDEO_ENABLED', True)
    monkeypatch.setattr(main, '_active_aicc_job_exists', lambda: False)
    monkeypatch.setattr(main, '_check_review_upload_space', lambda *args, **kwargs: None)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_UPLOAD_ROOT', tmp_path)
    return TestClient(main.app)


@pytest.fixture
def spools(monkeypatch):
    opened = []

    def track(*args, **kwargs):
        spool = tempfile.SpooledTemporaryFile(*args, **kwargs)
        opened.append(spool)
        return spool

    monkeypatch.setattr(starlette.formparsers, 'SpooledTemporaryFile', track)
    return opened


@pytest.mark.parametrize('parts,detail', [
    ([('file', ('one.avi', b'one')), ('extra', ('two.avi', b'two'))], 'Too many files'),
    ([('file', ('one.avi', b'one')), ('camera_id', (None, 'HN-001')),
      ('preserve_real_timing', (None, 'true')), ('extra', (None, 'extra'))], 'Too many fields'),
    ([('file', ('one.avi', b'one')), ('camera_id', (None, 'x' * 4097))], 'Part exceeded'),
])
def test_excess_parts_reject_before_enqueue_and_close_parser_files(
        review_client, monkeypatch, spools, parts, detail):
    def enqueue(*args, **kwargs):
        pytest.fail('Multipart limits must reject before CPU admission')

    monkeypatch.setattr(main, '_enqueue_video_review_job', enqueue)
    response = review_client.post('/api/scenarios/video-review', files=parts)
    assert response.status_code == 400, response.text
    assert detail in response.json()['detail']
    assert spools and all(spool.closed for spool in spools)
    assert not main.PUBLIC_UPLOAD_GATE.inflight


@pytest.mark.parametrize('content_type', ['application/x-www-form-urlencoded', 'application/octet-stream', ''])
def test_non_multipart_rejects_before_reading_body(review_client, content_type):
    async def run():
        messages = []
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
                 'http_version': '1.1', 'scheme': 'http', 'method': 'POST',
                 'path': '/api/scenarios/video-review', 'raw_path': b'/api/scenarios/video-review',
                 'query_string': b'', 'root_path': '',
                 'server': ('localhost', 80), 'client': ('127.0.0.1', 1234),
                 'headers': [(b'content-type', content_type.encode())] if content_type else []}

        async def receive():
            pytest.fail('Unsupported content types must reject without reading the request body')

        async def send(message):
            messages.append(message)

        await main.app(scope, receive, send)
        assert messages[0]['status'] == 415
        assert not main.PUBLIC_UPLOAD_GATE.inflight

    asyncio.run(run())


def test_one_file_with_camera_and_timing_still_enqueues_and_closes_form(
        review_client, monkeypatch, spools):
    calls = []
    monkeypatch.setattr(main, '_assert_camera_exists', lambda camera: calls.append(camera))
    monkeypatch.setattr(main, '_validate_review_upload', lambda path: None)

    def enqueue(path, filename, camera_id, preserve_real_timing, cleanup):
        assert path.read_bytes() == b'video fixture'
        assert filename == 'one.avi'
        assert camera_id == 'HN-001'
        assert preserve_real_timing is True
        assert cleanup is True
        path.unlink()
        main.ACTIVE_REVIEW_UPLOADS.discard(path)
        return {'job_id': 'a' * 32, 'status': 'queued'}

    monkeypatch.setattr(main, '_enqueue_video_review_job', enqueue)
    response = review_client.post('/api/scenarios/video-review', files={
        'file': ('one.avi', b'video fixture', 'video/x-msvideo'),
        'camera_id': (None, 'HN-001'), 'preserve_real_timing': (None, 'true')})
    assert response.status_code == 202, response.text
    assert response.json()['job_id'] == 'a' * 32
    assert calls == ['HN-001']
    assert spools and all(spool.closed for spool in spools)
    assert not main.PUBLIC_UPLOAD_GATE.inflight


@pytest.mark.parametrize('failure', ['validation', 'handler'])
def test_form_file_closes_when_validation_or_handler_fails(review_client, monkeypatch, spools, failure):
    parts = {'file': ('bad.txt', b'fixture')}
    if failure == 'validation':
        parts['preserve_real_timing'] = (None, 'not-a-boolean')
    response = review_client.post('/api/scenarios/video-review', files=parts)
    assert response.status_code == (422 if failure == 'validation' else 400)
    assert spools and all(spool.closed for spool in spools)
    assert not main.PUBLIC_UPLOAD_GATE.inflight
