"""Exact public byte boundaries, bounded copies, and terminal-worker recovery."""
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import io
import json
import threading
import time
from types import SimpleNamespace

import cv2
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
import numpy as np
import pytest

import app.main as main


MIB = 1024 * 1024
LIMIT = 500 * MIB


@pytest.fixture
def review_state(monkeypatch, tmp_path):
    uploads, outputs = tmp_path / 'uploads', tmp_path / 'outputs'
    uploads.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(main, 'VIDEO_REVIEW_UPLOAD_ROOT', uploads)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_ROOT', outputs)
    monkeypatch.setattr(main, 'VIDEO_REVIEW_JOBS', {})
    monkeypatch.setattr(main, 'AICC_JOBS', {})
    monkeypatch.setattr(main, 'ACTIVE_REVIEW_UPLOADS', set())
    monkeypatch.setattr(main, 'PUBLIC_DEMO_MODE', True)
    monkeypatch.setattr(main, 'PUBLIC_VIDEO_REVIEW_ENABLED', True)
    return uploads, outputs


class GeneratedUpload:
    """Generate one bounded chunk repeatedly, without allocating a video-sized buffer."""
    def __init__(self, total):
        self.remaining = total
        self.chunk = b'x' * MIB
        self.requests = []

    async def read(self, size):
        assert 0 < size <= MIB, 'Upload must request bounded chunks'
        self.requests.append(size)
        length = min(size, self.remaining)
        self.remaining -= length
        return self.chunk if length == MIB else self.chunk[:length]


class CountingSink:
    def __init__(self):
        self.bytes_written = 0
        self.largest_write = 0
        self.event_loop_thread = threading.get_ident()

    def write(self, chunk):
        assert threading.get_ident() != self.event_loop_thread
        self.bytes_written += len(chunk)
        self.largest_write = max(self.largest_write, len(chunk))


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_500_mib_stream_boundary_without_full_file_allocation(delta):
    async def run():
        upload = GeneratedUpload(LIMIT + delta)
        sink = CountingSink()
        if delta > 0:
            with pytest.raises(HTTPException) as error:
                await main._copy_review_upload(upload, sink, LIMIT)
            assert error.value.status_code == 413
            assert '500 MiB' in error.value.detail
            assert sink.bytes_written == LIMIT, 'Oversized bytes must never reach disk'
        else:
            assert await main._copy_review_upload(upload, sink, LIMIT) == LIMIT + delta
            assert sink.bytes_written == LIMIT + delta
        assert upload.remaining == 0
        assert sink.largest_write <= MIB
        assert len(upload.requests) >= 500

    asyncio.run(run())


def test_disk_headroom_accounts_for_spool_and_copy_on_shared_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'VIDEO_REVIEW_UPLOAD_ROOT', tmp_path)
    monkeypatch.setattr(main.tempfile, 'gettempdir', lambda: str(tmp_path))
    required = 2 * LIMIT + main.VIDEO_REVIEW_DISK_RESERVE_BYTES
    free = required
    monkeypatch.setattr(main.shutil, 'disk_usage', lambda path: SimpleNamespace(free=free))
    main._check_review_upload_space(LIMIT, include_spool=True)
    free -= 1
    with pytest.raises(HTTPException) as error:
        main._check_review_upload_space(LIMIT, include_spool=True)
    assert error.value.status_code == 507
    # The parser spool already exists when the endpoint starts copying.
    free = LIMIT + main.VIDEO_REVIEW_DISK_RESERVE_BYTES
    main._check_review_upload_space(LIMIT, include_spool=False)


def test_disk_headroom_checks_separate_filesystems(monkeypatch):
    upload_disk = SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=1))
    spool_disk = SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=2))
    monkeypatch.setattr(main, 'VIDEO_REVIEW_UPLOAD_ROOT', upload_disk)
    monkeypatch.setattr(main, 'Path', lambda raw: spool_disk)
    required = LIMIT + main.VIDEO_REVIEW_DISK_RESERVE_BYTES
    monkeypatch.setattr(main.shutil, 'disk_usage', lambda path: SimpleNamespace(
        free=required if path is upload_disk else required - 1))
    with pytest.raises(HTTPException) as error:
        main._check_review_upload_space(LIMIT, include_spool=True)
    assert error.value.status_code == 507


def test_known_oversized_file_is_rejected_before_copy_and_closed(review_state, monkeypatch):
    file = UploadFile(io.BytesIO(b'not read'), filename='large.avi', size=LIMIT + 1)
    monkeypatch.setattr(main, '_copy_review_upload', lambda *args: pytest.fail('Must reject before copying'))
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.create_video_review(file=file, camera_id=None, preserve_real_timing=False))
    assert error.value.status_code == 413
    assert file.file.closed
    assert not list(review_state[0].iterdir())


def test_parser_spool_is_closed_before_validation_and_enqueue(review_state, monkeypatch):
    file = UploadFile(io.BytesIO(b'bounded fixture'), filename='small.avi', size=15)
    monkeypatch.setattr(main, '_check_review_upload_space', lambda *args, **kwargs: None)
    def validate(path):
        assert file.file.closed
        assert path.read_bytes() == b'bounded fixture'
    def enqueue(path, *args, **kwargs):
        assert file.file.closed
        path.unlink()
        main.ACTIVE_REVIEW_UPLOADS.discard(path)
        return {'job_id': 'a' * 32, 'status': 'queued'}
    monkeypatch.setattr(main, '_validate_review_upload', validate)
    monkeypatch.setattr(main, '_enqueue_video_review_job', enqueue)
    accepted = asyncio.run(main.create_video_review(file=file, camera_id=None, preserve_real_timing=False))
    assert accepted['status'] == 'queued'
    assert not list(review_state[0].iterdir())


def test_canceled_copy_closes_spool_and_removes_owned_temporary_file(review_state, monkeypatch):
    monkeypatch.setattr(main, '_check_review_upload_space', lambda *args, **kwargs: None)
    async def run():
        reading = asyncio.Event()
        class InterruptedUpload:
            filename = 'interrupted.avi'
            size = None
            closed = False
            reads = 0
            async def read(self, size):
                self.reads += 1
                if self.reads == 1:
                    return b'small first chunk'
                reading.set()
                await asyncio.Event().wait()
            async def close(self):
                self.closed = True
        file = InterruptedUpload()
        task = asyncio.create_task(main.create_video_review(file=file, camera_id=None, preserve_real_timing=False))
        await asyncio.wait_for(reading.wait(), 2)
        assert len(main.ACTIVE_REVIEW_UPLOADS) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert file.closed
        assert not main.ACTIVE_REVIEW_UPLOADS
        assert not list(review_state[0].iterdir())
    asyncio.run(run())


@pytest.mark.parametrize('completion', ['cancel', 'exception', 'return'])
def test_terminal_future_cannot_leave_job_queued_or_block_next_upload(review_state, monkeypatch, completion):
    future = Future()
    monkeypatch.setattr(main.AICC_EXECUTOR, 'submit', lambda *args: future)
    path = review_state[0] / 'traffic_review_fixture.avi'
    path.write_bytes(b'fixture')
    main.ACTIVE_REVIEW_UPLOADS.add(path)
    queued = main._enqueue_video_review_job(path, path.name, None, False, cleanup=True)
    assert main._active_aicc_job_exists()
    if completion == 'cancel':
        assert future.cancel()
    elif completion == 'exception':
        future.set_exception(RuntimeError('unexpected worker exit'))
    else:
        future.set_result(None)  # A broken worker returned without setting terminal status.
    assert not main._active_aicc_job_exists()
    job = main.video_review_status(queued['job_id'])
    assert job['status'] == 'failed'
    assert job['finished_at']
    assert not path.exists()
    assert not main.ACTIVE_REVIEW_UPLOADS
    assert not main.VIDEO_REVIEW_FUTURES


@pytest.mark.parametrize('status', ['queued', 'running'])
def test_old_live_future_remains_busy_and_diagnostics_are_private(review_state, status):
    job_id, filename = 'b' * 32, 'private-camera-location.avi'
    future = Future()
    if status == 'running':
        assert future.set_running_or_notify_cancel()
    main.VIDEO_REVIEW_JOBS[job_id] = {
        'status': status, 'filename': filename, 'created_epoch': 0, 'progress_epoch': 0,
        'phase': 'loading_models', 'progress_percent': 0}
    main.VIDEO_REVIEW_FUTURES[job_id] = future
    assert main._active_aicc_job_exists()
    activity = main._video_review_activity()
    assert activity['analysis_in_progress']
    assert activity['active_jobs'][0]['phase'] == 'loading_models'
    assert activity['active_jobs'][0]['age_seconds'] > 3600
    assert job_id not in json.dumps(activity)
    assert filename not in json.dumps(activity)
    assert main.VIDEO_REVIEW_JOBS[job_id]['status'] == status
    assert main.VIDEO_REVIEW_FUTURES[job_id] is future


@pytest.mark.parametrize('status', ['succeeded', 'failed'])
def test_completed_terminal_status_is_preserved_and_does_not_hold_cpu(review_state, status):
    job_id = 'c' * 32
    main.VIDEO_REVIEW_JOBS[job_id] = {'status': status, 'result': {'preserved': True}}
    future = Future()
    future.set_result(None)
    main.VIDEO_REVIEW_FUTURES[job_id] = future
    assert not main._active_aicc_job_exists()
    assert main.VIDEO_REVIEW_JOBS[job_id]['status'] == status
    assert main.VIDEO_REVIEW_JOBS[job_id]['result'] == {'preserved': True}


def test_small_upload_returns_job_then_polled_result_and_cleans_tempfile(review_state, monkeypatch, tmp_path):
    source = tmp_path / 'small.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (96, 64))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(main, '_load_video_review_models', lambda: None)
    monkeypatch.setattr(main.aicc_runtime, 'load', lambda: {})
    def analyze(path, bundle, output_dir, **kwargs):
        assert path.is_file()
        output_dir.mkdir()
        (output_dir / 'evidence.csv').write_text('fixture,value\nsynthetic,1\n')
        kwargs['progress_callback'](50, 1, 2)
        return {'frames': [], 'fixture': 'synthetic lifecycle only'}
    monkeypatch.setattr(main, 'analyze_video', analyze)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(main, 'AICC_EXECUTOR', executor)
        with TestClient(main.app) as client:
            accepted = client.post('/api/scenarios/video-review', files={
                'file': ('small.avi', source.read_bytes(), 'video/x-msvideo')})
            assert accepted.status_code == 202, accepted.text
            job = accepted.json()
            assert len(job['job_id']) == 32
            deadline = time.monotonic() + 5
            while True:
                response = client.get(job['poll_url'])
                assert response.status_code == 200
                result = response.json()
                if result['status'] in {'succeeded', 'failed'}:
                    break
                assert time.monotonic() < deadline, 'Job failed to reach terminal status'
                time.sleep(0.01)
            assert result['status'] == 'succeeded', result
            assert result['result']['fixture'] == 'synthetic lifecycle only'
            assert result['progress_percent'] == 100
            csv = client.get(result['result']['evidence_csv_url'])
            assert csv.status_code == 200
            assert 'synthetic,1' in csv.text
    assert not main._active_aicc_job_exists()
    assert not main.ACTIVE_REVIEW_UPLOADS
    assert not list(review_state[0].iterdir())
