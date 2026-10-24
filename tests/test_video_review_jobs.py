"""Small local fixtures; model execution is stubbed only in worker unit tests."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
import app.main as main

client = TestClient(main.app)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setattr(main, 'DATA_DIR', data)
    monkeypatch.setattr(main, 'CAMERA_FEED_DIR', data / 'camera_feeds')
    monkeypatch.setattr(main, 'VIDEO_REVIEW_ROOT', data / 'video_reviews')
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {})
    monkeypatch.setattr(main, 'AICC_JOBS', {})
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', False)
    monkeypatch.setattr(main, 'AICC_VIDEO_ENABLED', True)
    monkeypatch.setattr(main.aicc_runtime, 'status', lambda: {'missing_files': []})
    return data


@pytest.fixture
def clip(sandbox):
    path = sandbox / 'nested' / 'small.avi'
    path.parent.mkdir()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (96, 64))
    if not writer.isOpened():
        pytest.skip('MJPG encoder unavailable')
    for _ in range(5):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    return path


def test_empty_list_and_missing_video(sandbox):
    assert client.get('/api/aicc/data-videos').json()['items'] == []
    response = client.post('/api/scenarios/video-review-from-data', json={'path': 'gone.mp4'})
    assert response.status_code == 404
    assert 'Tải lại danh sách' in response.json()['detail']


def test_listing_nested_and_excluded_files(sandbox, clip):
    (sandbox / 'empty.mp4').touch()
    (sandbox / 'outside.mp4').symlink_to(sandbox.parent / 'elsewhere.mp4')
    for folder in ('video_reviews', 'camera_feeds'):
        target = sandbox / folder
        target.mkdir()
        (target / 'internal.avi').write_bytes(clip.read_bytes())
    payload = client.get('/api/aicc/data-videos').json()
    assert payload['total'] == 1
    assert payload['items'][0]['path'] == 'nested/small.avi'


@pytest.mark.parametrize('path,code', [('../outside.mp4', 400), ('notes.txt', 400),
    ('video_reviews/output.mp4', 400), ('', 422)])
def test_invalid_path(sandbox, path, code):
    assert client.post('/api/scenarios/video-review-from-data', json={'path': path}).status_code == code


def test_symlink_escape(sandbox):
    outside = sandbox.parent / 'outside.mp4'
    outside.write_bytes(b'not used as video')
    (sandbox / 'link.mp4').symlink_to(outside)
    assert client.post('/api/scenarios/video-review-from-data', json={'path': 'link.mp4'}).status_code == 400


def test_size_limit(sandbox, clip, monkeypatch):
    monkeypatch.setattr(main, 'MAX_VIDEO_REVIEW_BYTES', 1)
    assert client.get('/api/aicc/data-videos').json()['items'] == []
    assert client.post('/api/scenarios/video-review-from-data', json={'path': 'nested/small.avi'}).status_code == 413


def test_upload_and_data_queue_independently(sandbox, clip, monkeypatch):
    submitted = []
    monkeypatch.setattr(main.AICC_EXECUTOR, 'submit', lambda *args: submitted.append(args))
    for upload in (False, True):
        if upload:
            response = client.post('/api/scenarios/video-review', files={'file': ('chosen.avi', clip.read_bytes(), 'video/x-msvideo')})
        else:
            response = client.post('/api/scenarios/video-review-from-data', json={'path': 'nested/small.avi'})
        assert response.status_code == 202, response.text
        job = response.json()
        assert len(job['job_id']) == 32
        assert client.get(job['poll_url']).json()['status'] == 'queued'
        args = submitted[-1]
        assert args[-1] is upload  # only uploads are temporary and cleaned by worker
        busy = client.post('/api/scenarios/video-review-from-data', json={'path': 'nested/small.avi'})
        assert busy.status_code == 409
        main.VIDEO_REVIEW_JOBS[job['job_id']]['status'] = 'failed'
        if upload:
            args[2].unlink()
    assert submitted[0][2] == clip.resolve()
    assert submitted[1][3] == 'chosen.avi'


@pytest.mark.parametrize('fail', [False, True])
def test_worker_terminal_status_and_cleanup(sandbox, clip, monkeypatch, fail):
    job_id = 'a' * 32
    main.VIDEO_REVIEW_JOBS[job_id] = {'status': 'queued'}
    monkeypatch.setattr(main.aicc_runtime, 'load', lambda: {})
    def analyze(*args, **kwargs):
        kwargs['progress_callback'](40, 2, 5)
        assert main.VIDEO_REVIEW_JOBS[job_id]['progress_percent'] == 40
        if fail:
            raise RuntimeError('test decoder failure')
        return {'frames': [{'filename': 'frame_001.jpg'}]}
    monkeypatch.setattr(main, 'analyze_video', analyze)
    main._run_video_review_job(job_id, clip, clip.name, None, False, True)
    job = client.get(f'/api/scenarios/video-reviews/{job_id}').json()
    assert job['status'] == ('failed' if fail else 'succeeded')
    assert job['finished_at']
    assert not clip.exists()
    if fail:
        assert 'test decoder failure' in job['error']
    else:
        assert job['result']['frames'][0]['url'].endswith('/frames/frame_001.jpg')
        assert job['progress_percent'] == 100


def test_submission_failure_does_not_poison_cpu_slot(sandbox, clip, monkeypatch):
    def reject(*args):
        raise RuntimeError('executor stopped')
    monkeypatch.setattr(main.AICC_EXECUTOR, 'submit', reject)
    response = client.post('/api/scenarios/video-review-from-data', json={'path': 'nested/small.avi'})
    assert response.status_code == 503
    assert not main._active_aicc_job_exists()
    assert all(job['status'] == 'failed' for job in main.VIDEO_REVIEW_JOBS.values())
    monkeypatch.setattr(main.AICC_EXECUTOR, 'submit', lambda *args: None)
    assert client.post('/api/scenarios/video-review-from-data', json={'path': 'nested/small.avi'}).status_code == 202


def test_unknown_job_and_no_stale_cache(sandbox):
    assert client.get('/api/scenarios/video-reviews/' + 'b' * 32).status_code == 404
    for path in ('/api/aicc/data-videos', '/static/app.js'):
        assert client.get(path).headers['cache-control'] == 'no-store'
