"""Public ingress, non-blocking job admission, retention, and administrative boundaries."""
import asyncio
from concurrent.futures import Future
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest
from starlette.exceptions import HTTPException

import app.main as main
from app.public_upload import UploadGate, PublicUploadMiddleware

client = TestClient(main.app)


def test_global_quota_cannot_be_reset_with_forged_forwarded_headers(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, 'PUBLIC_UPLOAD_GATE', UploadGate(attempts=2))
    for index in range(3):
        response = client.post('/api/scenarios/video-review', files={'file': ('bad.txt', b'x')},
                               headers={'X-Forwarded-For': f'1.2.3.{index}'})
        assert response.status_code == (429 if index == 2 else 400)


def test_busy_rejected_before_receiving_file(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, '_active_aicc_job_exists', lambda: True)
    response = client.post('/api/scenarios/video-review', content=b'not parsed as multipart')
    assert response.status_code == 409
    assert response.json()['detail'] == 'Hệ thống đang phân tích một video khác. Vui lòng thử lại sau'
    assert response.headers['retry-after']


def test_public_content_length_is_bounded_before_multipart(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    response = client.post('/api/scenarios/video-review', content=b'x', headers={'Content-Length': str(102*1024*1024)})
    assert response.status_code == 413


def test_chunked_body_limit_and_idle_timeout():
    async def run():
        async def consume(scope, receive, send):
            while True:
                if not (await receive()).get('more_body'):
                    break
        gate = UploadGate()
        middleware = PublicUploadMiddleware(consume, lambda: True, lambda: True, lambda: 0,
                                            lambda: False, lambda: gate)
        scope = {'type': 'http', 'method': 'POST', 'path': '/api/scenarios/video-review', 'headers': []}
        async def receive():
            return {'type':'http.request','body':b'x'*(1024*1024+1),'more_body':False}
        async def send(message):
            pass
        with pytest.raises(HTTPException) as error:
            await middleware(scope, receive, send)
        assert error.value.status_code == 413
        assert not gate.inflight
    asyncio.run(run())


def test_admission_does_not_load_models_on_request_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {})
    monkeypatch.setattr(main, 'AICC_JOBS', {})
    def wrong_thread():
        pytest.fail('Model loading must happen in the worker, after HTTP 202')
    monkeypatch.setattr(main, '_load_video_review_models', wrong_thread)
    future = Future()
    monkeypatch.setattr(main.AICC_EXECUTOR, 'submit', lambda *args: future)
    path = tmp_path/'fixture.mp4'
    path.write_bytes(b'not opened by admission')
    job = main._enqueue_video_review_job(path, path.name, None, False, True)
    assert job['status'] == 'queued'
    assert main.VIDEO_REVIEW_JOBS[job['job_id']]['status'] == 'queued'
    future.cancel()
    assert main.VIDEO_REVIEW_JOBS[job['job_id']]['status'] == 'failed'
    assert not path.exists()
    assert not main._active_aicc_job_exists()


def test_expired_outputs_and_restart_orphans_are_cleaned(monkeypatch, tmp_path):
    root = tmp_path/'reviews';root.mkdir()
    monkeypatch.setattr(main, 'VIDEO_REVIEW_ROOT', root)
    old_id, active_id, orphan_id = 'a'*32, 'b'*32, 'c'*32
    for name in (old_id, active_id, orphan_id, 'personal-folder'):
        (root/name).mkdir()
        os.utime(root/name, (0,0))
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {
        old_id:{'status':'failed','finished_epoch':0}, active_id:{'status':'running'}})
    main._cleanup_video_reviews()
    assert not (root/old_id).exists()
    assert not (root/orphan_id).exists()
    assert (root/active_id).exists() and (root/'personal-folder').exists()


def test_public_opt_in_does_not_open_other_writes(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, 'AICC_VIDEO_ENABLED', False)
    for path in ('/api/aicc/load', '/api/models/reload', '/api/scenarios/video-review-from-data'):
        assert client.post(path, json={'path':'clip.mp4'}).status_code == 403
    assert client.post('/api/aicc/predict-video', files={'file':('clip.mp4',b'x')}).status_code == 403
    assert client.post('/api/cameras/HN-001/feed', files={'file':('clip.mp4',b'x')}).status_code == 403


def test_public_ui_has_one_analysis_flow():
    html = client.get('/').text
    assert 'id="scenarioVideoFile"' in html
    for removed in ('scenarioDataVideo', 'scenarioAnalyzeDataBtn', 'aiccVideoFile', 'aiccAnalyzeBtn', 'cameraFeedDataSelect'):
        assert f'id="{removed}"' not in html
    scenario = html.split('id="page-video"',1)[1].split('id="page-settings"',1)[0]
    assert scenario.count('type="file"') == 1
    assert 'CPU' not in scenario and 'data/' not in scenario
    assert 'scenarioPreserveTiming' in scenario and 'scenarioExportBtn' in scenario


def test_idle_upload_releases_admission_slot():
    async def run():
        async def consume(scope, receive, send):
            await receive()
        async def stalled():
            await asyncio.sleep(10)
        async def send(message):
            pass
        gate = UploadGate()
        middleware = PublicUploadMiddleware(consume, lambda: True, lambda: True, lambda: 100,
                                            lambda: False, lambda: gate, idle_timeout=0.001)
        scope = {'type':'http','method':'POST','path':'/api/scenarios/video-review','headers':[]}
        with pytest.raises(HTTPException) as error:
            await middleware(scope, stalled, send)
        assert error.value.status_code == 408
        assert not gate.inflight
    asyncio.run(run())


def test_real_multipart_without_content_length_is_still_bounded(monkeypatch):
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, 'MAX_PUBLIC_VIDEO_REVIEW_BYTES', 1)
    def chunks():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
        yield b'x' * (1024*1024+10)
        yield b'\r\n--boundary--\r\n'
    response = client.post('/api/scenarios/video-review', content=chunks(), headers={'Content-Type':'multipart/form-data; boundary=boundary'})
    assert response.status_code == 413, response.text
    assert not main.PUBLIC_UPLOAD_GATE.inflight


def test_temporary_upload_retention_skips_active_files(monkeypatch, tmp_path):
    root = tmp_path/'reviews';root.mkdir()
    uploads = tmp_path/'uploads';uploads.mkdir()
    old, active = uploads/'traffic_review_old.mp4', uploads/'traffic_review_active.mp4'
    for path in (old, active):
        path.write_bytes(b'fixture');os.utime(path,(0,0))
    monkeypatch.setattr(main, 'VIDEO_REVIEW_ROOT', root)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_UPLOAD_ROOT', uploads)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {})
    monkeypatch.setattr(main, 'ACTIVE_REVIEW_UPLOADS', {active})
    main._cleanup_video_reviews()
    assert not old.exists() and active.exists()


def test_cold_model_loading_does_not_hold_http_acceptance_or_health(monkeypatch, tmp_path):
    import threading
    import cv2
    import numpy as np
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {})
    monkeypatch.setattr(main, 'AICC_JOBS', {})
    def slow_load():
        entered.set()
        release.wait(5)
    monkeypatch.setattr(main, '_load_video_review_models', slow_load)
    monkeypatch.setattr(main.aicc_runtime, 'load', lambda: {})
    monkeypatch.setattr(main, 'analyze_video', lambda *args,**kwargs: {'frames':[]})
    source=tmp_path/'small.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),5,(96,64))
    assert writer.isOpened()
    writer.write(np.zeros((64,96,3),dtype=np.uint8));writer.release()
    responses=[]
    accepted=threading.Event()
    def upload():
        responses.append(client.post('/api/scenarios/video-review',files={'file':('small.avi',source.read_bytes())}))
        accepted.set()
    sender=threading.Thread(target=upload);sender.start()
    try:
        assert entered.wait(2)
        assert accepted.wait(2), 'HTTP 202 waited for cold model loading'
        assert responses[0].status_code == 202
        assert client.get('/api/health').status_code == 200
        assert client.get(responses[0].json()['poll_url']).json()['status'] == 'running'
    finally:
        release.set();sender.join(timeout=5)
        deadline=time.monotonic()+5
        while main._active_aicc_job_exists() and time.monotonic()<deadline:
            time.sleep(.01)


def test_expired_results_cannot_be_read_even_if_files_remain(monkeypatch, tmp_path):
    job_id='d'*32
    root=tmp_path/'reviews';(root/job_id/'frames').mkdir(parents=True)
    (root/job_id/'frames'/'frame_001.jpg').write_bytes(b'fixture')
    (root/job_id/'evidence.csv').write_text('header\n')
    monkeypatch.setattr(main,'VIDEO_REVIEW_ROOT',root)
    monkeypatch.setattr(main,'VIDEO_REVIEW_JOBS',{})
    assert client.get(f'/api/scenarios/video-reviews/{job_id}/frames/frame_001.jpg').status_code == 404
    assert client.get(f'/api/scenarios/video-reviews/{job_id}/csv').status_code == 404
