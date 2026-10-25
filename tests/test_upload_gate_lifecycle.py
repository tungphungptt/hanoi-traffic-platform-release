"""Upload admission follows HTTP lifetime without unlocking a newer request."""
import asyncio
import json
import socket

import httpx2
import pytest
import uvicorn
from starlette.exceptions import HTTPException

from app.public_upload import PublicUploadMiddleware, UploadGate


def scope(request_id='request'):
    return {'type': 'http', 'method': 'POST', 'path': '/api/scenarios/video-review',
            'headers': [], 'request_id': request_id}


async def receive_body():
    return {'type': 'http.request', 'body': b'x', 'more_body': False}


async def response(send, status=400):
    await send({'type': 'http.response.start', 'status': status, 'headers': []})
    await send({'type': 'http.response.body', 'body': b'invalid video', 'more_body': False})


def middleware(app, gate, busy=lambda: False, **kwargs):
    return PublicUploadMiddleware(app, lambda: True, lambda: True, lambda: 1024,
                                  busy, lambda: gate, **kwargs)


def test_completed_response_releases_slot_without_old_cleanup_unlocking_new_upload():
    async def run():
        gate = UploadGate()
        first_response = asyncio.Event()
        first_cleanup = asyncio.Event()
        second_entered = asyncio.Event()
        second_finish = asyncio.Event()
        messages = {'first': [], 'second': [], 'third': []}

        async def app(request, receive, send):
            if request['request_id'] == 'first':
                await response(send)
                first_response.set()
                await first_cleanup.wait()
            else:
                second_entered.set()
                await second_finish.wait()
                await response(send)

        wrapped = middleware(app, gate)

        async def invoke(request_id):
            async def send(message):
                messages[request_id].append(message)
            await wrapped(scope(request_id), receive_body, send)

        first = asyncio.create_task(invoke('first'))
        second = None
        try:
            await asyncio.wait_for(first_response.wait(), 1)
            assert not gate.inflight, 'HTTP completed but post-response cleanup retained admission'
            second = asyncio.create_task(invoke('second'))
            await asyncio.sleep(0)
            assert second_entered.is_set(), messages['second']
            assert gate.inflight
            first_cleanup.set()
            await first
            assert gate.inflight, 'An older request must not release the current admission'
            await invoke('third')
            assert messages['third'][0]['status'] == 409
            second_finish.set()
            await second
            assert not gate.inflight
        finally:
            first_cleanup.set()
            second_finish.set()
            await asyncio.gather(first, *([second] if second is not None else []))

    asyncio.run(run())


def test_real_uvicorn_allows_next_upload_after_http_response_before_cleanup():
    async def run():
        gate = UploadGate()
        cleanup = asyncio.Event()
        completed = 0

        async def app(request, receive, send):
            nonlocal completed
            await receive()
            await response(send)
            completed += 1
            await cleanup.wait()

        server = uvicorn.Server(uvicorn.Config(middleware(app, gate), lifespan='off',
                                               log_level='critical', access_log=False))
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        address = f'http://127.0.0.1:{listener.getsockname()[1]}'
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async def ready():
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(ready(), 3)
            async with httpx2.AsyncClient(base_url=address, timeout=3, trust_env=False) as client:
                first = await client.post('/api/scenarios/video-review', content=b'invalid')
                second = await client.post('/api/scenarios/video-review', content=b'invalid')
            assert first.status_code == 400
            assert second.status_code == 400, second.text
            assert completed == 2
            assert not gate.inflight
        finally:
            cleanup.set()
            server.should_exit = True
            await asyncio.wait_for(serving, 3)
            listener.close()

    asyncio.run(run())


def test_real_cpu_job_remains_busy_after_ingress_response_finishes():
    async def run():
        gate = UploadGate()
        running = False

        async def app(request, receive, send):
            nonlocal running
            running = True
            await response(send, status=202)

        wrapped = middleware(app, gate, busy=lambda: running)
        messages = []

        async def send(message):
            messages.append(message)

        await wrapped(scope(), receive_body, send)
        assert not gate.inflight
        messages.clear()

        async def never_receive():
            pytest.fail('Busy CPU jobs must reject before accepting another body')

        await wrapped(scope(), never_receive, send)
        assert messages[0]['status'] == 409
        assert json.loads(messages[1]['body'])['busy_reason'] == 'analysis_in_progress'
        assert not gate.inflight

    asyncio.run(run())


def test_streaming_response_keeps_admission_until_final_body_is_sent():
    async def run():
        gate = UploadGate()
        final_started = asyncio.Event()
        finish_send = asyncio.Event()

        async def app(request, receive, send):
            await send({'type': 'http.response.start', 'status': 400, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'partial', 'more_body': True})
            assert gate.inflight
            await send({'type': 'http.response.body', 'body': b'final'})

        async def slow_send(message):
            if message['type'] == 'http.response.body' and not message.get('more_body'):
                final_started.set()
                await finish_send.wait()

        request = asyncio.create_task(middleware(app, gate)(scope(), receive_body, slow_send))
        try:
            await asyncio.wait_for(final_started.wait(), 1)
            assert gate.inflight
            lease, rejected = gate.admit(lambda: False)
            assert lease is None and rejected[2] == 'upload_in_progress'
            snapshot = gate.snapshot()
            assert set(snapshot) == {'upload_in_progress', 'upload_age_seconds'}
            assert snapshot['upload_in_progress'] is True
            assert snapshot['upload_age_seconds'] >= 0
        finally:
            finish_send.set()
            await request
        assert gate.snapshot() == {'upload_in_progress': False, 'upload_age_seconds': None}

    asyncio.run(run())


def test_disconnect_releases_slot_before_inner_cleanup_returns():
    async def run():
        gate = UploadGate()
        disconnected = asyncio.Event()
        cleanup = asyncio.Event()

        async def app(request, receive, send):
            assert (await receive())['type'] == 'http.disconnect'
            disconnected.set()
            await cleanup.wait()

        async def receive_disconnect():
            return {'type': 'http.disconnect'}

        async def send(message):
            pytest.fail('Disconnected upload must not need a response to release admission')

        request = asyncio.create_task(middleware(app, gate)(scope(), receive_disconnect, send))
        try:
            await asyncio.wait_for(disconnected.wait(), 1)
            assert not gate.inflight
        finally:
            cleanup.set()
            await request

    asyncio.run(run())


def test_cancellation_releases_slot_and_receive_stays_in_original_asgi_task():
    async def run():
        gate = UploadGate()
        entered = asyncio.Event()
        app_task = None

        async def app(request, receive, send):
            nonlocal app_task
            app_task = asyncio.current_task()
            await receive()

        async def stalled_receive():
            assert asyncio.current_task() is app_task
            entered.set()
            await asyncio.Event().wait()

        async def send(message):
            pass

        request = asyncio.create_task(middleware(app, gate)(scope(), stalled_receive, send))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert gate.inflight
        finally:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        assert not gate.inflight
        # Cancellation must not poison the next request's admission.
        await middleware(lambda_app, gate)(scope(), receive_body, send)
        assert not gate.inflight

    async def lambda_app(request, receive, send):
        await response(send)

    asyncio.run(run())


@pytest.mark.parametrize('failure', ['idle_timeout', 'total_timeout', 'body_limit',
                                     'receive_error', 'send_error', 'app_error'])
def test_failure_releases_admission(failure):
    async def run():
        gate = UploadGate()

        async def app(request, receive, send):
            if failure == 'app_error':
                raise RuntimeError('application failed')
            if failure == 'send_error':
                return await response(send)
            await receive()

        async def receive():
            if failure == 'idle_timeout':
                await asyncio.Event().wait()
            if failure == 'receive_error':
                raise ConnectionResetError('connection lost')
            return {'type': 'http.request', 'body': b'x' * (1024 * 1024 + 1025), 'more_body': True}

        async def send(message):
            raise OSError('connection lost while responding')

        expected = {'idle_timeout': HTTPException, 'total_timeout': HTTPException,
                    'body_limit': HTTPException, 'receive_error': ConnectionResetError,
                    'send_error': OSError, 'app_error': RuntimeError}[failure]
        wrapped = middleware(app, gate, idle_timeout=0.001,
                             total_timeout=0 if failure == 'total_timeout' else 600)
        with pytest.raises(expected) as caught:
            await wrapped(scope(), receive, send)
        if failure in ('idle_timeout', 'total_timeout'):
            assert caught.value.status_code == 408
        elif failure == 'body_limit':
            assert caught.value.status_code == 413
        assert gate.snapshot() == {'upload_in_progress': False, 'upload_age_seconds': None}

    asyncio.run(run())


@pytest.mark.parametrize('content_length', [None, 0, 17, 1024 * 1024 + 1024])
def test_storage_preflight_runs_under_lease_before_body_parser(content_length):
    async def run():
        gate = UploadGate()
        checked = []

        def storage_check(expected_bytes):
            assert gate.inflight
            checked.append(expected_bytes)

        async def app(request, receive, send):
            assert checked, 'Disk space must be checked before multipart parsing'
            await receive()
            await response(send)

        async def send(message):
            pass

        request = scope()
        if content_length is not None:
            request['headers'] = [(b'content-length', str(content_length).encode())]
        await middleware(app, gate, storage_check=storage_check)(request, receive_body, send)
        assert checked == [content_length if content_length is not None else 1024 * 1024 + 1024]
        assert not gate.inflight

    asyncio.run(run())


def test_insufficient_storage_returns_507_without_reading_body_and_releases_lease():
    async def run():
        gate = UploadGate()
        messages = []

        def storage_check(expected_bytes):
            assert gate.inflight
            raise HTTPException(507, 'Insufficient storage')

        async def app(request, receive, send):
            pytest.fail('Storage rejection must happen before multipart parsing')

        async def receive():
            pytest.fail('Storage rejection must not read the upload body')

        async def send(message):
            messages.append(message)

        await middleware(app, gate, storage_check=storage_check)(scope(), receive, send)
        assert messages[0]['status'] == 507
        assert json.loads(messages[1]['body'])['detail'] == 'Insufficient storage'
        assert not gate.inflight
        # Even the error response cannot hold the ingress slot if send fails.
        async def broken_send(message):
            raise OSError('connection lost')
        with pytest.raises(OSError):
            await middleware(app, gate, storage_check=storage_check)(scope(), receive, broken_send)
        assert not gate.inflight

    asyncio.run(run())


@pytest.mark.parametrize('rejection', ['busy', 'oversize', 'rate_limit'])
def test_rejected_request_does_not_run_storage_check(rejection):
    async def run():
        gate = UploadGate(attempts=0 if rejection == 'rate_limit' else 6)
        messages = []

        def storage_check(expected_bytes):
            pytest.fail('Already rejected requests must not check storage')

        async def app(request, receive, send):
            pytest.fail('Already rejected requests must not parse uploads')

        async def send(message):
            messages.append(message)

        request = scope()
        if rejection == 'oversize':
            request['headers'] = [(b'content-length', str(1024 * 1024 + 1025).encode())]
        await middleware(app, gate, busy=lambda: rejection == 'busy', storage_check=storage_check)(
            request, receive_body, send)
        assert messages[0]['status'] == {'busy': 409, 'oversize': 413, 'rate_limit': 429}[rejection]
        assert not gate.inflight

    asyncio.run(run())


def test_cancellation_after_completed_body_releases_pre_response_admission():
    async def run():
        gate = UploadGate()
        body_complete = asyncio.Event()

        async def app(request, receive, send):
            assert (await receive())['more_body'] is False
            body_complete.set()
            await asyncio.Event().wait()

        async def send(message):
            pytest.fail('Cancelled request has not produced a response')

        request = asyncio.create_task(middleware(app, gate)(scope(), receive_body, send))
        try:
            await asyncio.wait_for(body_complete.wait(), 1)
            assert gate.inflight
        finally:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        assert not gate.inflight

    asyncio.run(run())
