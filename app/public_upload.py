"""Bound public upload traffic before multipart parsing; never trust forwarded IP headers."""
import asyncio
from collections import deque
import threading
import time

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

BUSY_MESSAGE = "Hệ thống đang phân tích một video khác. Vui lòng thử lại sau"


class UploadGate:
    # A global limit deliberately avoids trusting X-Forwarded-For or a spoofable client IP.
    def __init__(self, attempts=6, window_seconds=60):
        self.attempts = attempts
        self.window_seconds = window_seconds
        self.starts = deque()
        self._lease = None
        self._started_at = None
        self.lock = threading.Lock()

    @property
    def inflight(self):
        with self.lock:
            return self._lease is not None

    def snapshot(self):
        """Expose ingress state without request identifiers or uploaded filenames."""
        with self.lock:
            return {
                'upload_in_progress': self._lease is not None,
                'upload_age_seconds': (round(max(0, time.monotonic() - self._started_at), 3)
                                       if self._started_at is not None else None),
            }

    def admit(self, busy):
        with self.lock:
            now = time.monotonic()
            while self.starts and now - self.starts[0] >= self.window_seconds:
                self.starts.popleft()
            if self._lease is not None:
                return None, (409, BUSY_MESSAGE, 'upload_in_progress')
            if busy():
                return None, (409, BUSY_MESSAGE, 'analysis_in_progress')
            if len(self.starts) >= self.attempts:
                return None, (429, "Có quá nhiều lượt tải video. Vui lòng thử lại sau một phút.", None)
            self.starts.append(now)
            self._lease = object()
            self._started_at = now
            return self._lease, None

    def release(self, lease):
        with self.lock:
            # A completed request may still be doing post-response cleanup after
            # the next upload has entered. Its finalizer only owns its own lease.
            if self._lease is lease:
                self._lease = None
                self._started_at = None


class PublicUploadMiddleware:
    def __init__(self, app, public_mode, enabled, max_bytes, busy, gate, idle_timeout=60,
                 total_timeout=1800, storage_check=None):
        self.app = app
        self.public_mode = public_mode
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.busy = busy
        self.gate = gate
        self.idle_timeout = idle_timeout
        self.total_timeout = total_timeout
        self.storage_check = storage_check

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope['method'] != 'POST'
                or scope['path'] != '/api/scenarios/video-review' or not self.public_mode()):
            return await self.app(scope, receive, send)
        if not self.enabled():
            return await JSONResponse({'detail': 'Tính năng tải video để phân tích hiện đang bị tắt.'}, 403)(scope, receive, send)
        # Allow bounded multipart overhead, while the endpoint enforces the exact file limit.
        body_limit = self.max_bytes() + 1024 * 1024
        length = dict(scope.get('headers', [])).get(b'content-length')
        if length:
            try:
                if int(length) < 0:
                    raise ValueError
                oversized = int(length) > body_limit
            except ValueError:
                return await JSONResponse({'detail': 'Kích thước yêu cầu không hợp lệ.'}, 400)(scope, receive, send)
            if oversized:
                return await JSONResponse({'detail': 'Video vượt giới hạn tải lên.'}, 413)(scope, receive, send)
        gate = self.gate()
        lease, rejected = gate.admit(self.busy)
        if rejected:
            code, detail, busy_reason = rejected
            payload = {'detail': detail}
            if busy_reason:
                payload['busy_reason'] = busy_reason
            return await JSONResponse(payload, code, headers={'Retry-After': '60' if code == 429 else '5'})(scope, receive, send)
        consumed = 0
        deadline = time.monotonic() + self.total_timeout

        async def bounded_receive():
            nonlocal consumed
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                gate.release(lease)
                raise HTTPException(408, 'Tải video quá thời gian cho phép. Vui lòng thử lại.')
            try:
                # Keep receive in this ASGI task; a child task changes cancellation
                # ownership for middleware and multipart parser cleanup.
                async with asyncio.timeout(min(self.idle_timeout, remaining)):
                    message = await receive()
            except asyncio.TimeoutError:
                gate.release(lease)
                raise HTTPException(408, 'Kết nối tải video bị gián đoạn. Vui lòng thử lại.') from None
            except BaseException:
                gate.release(lease)
                raise
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > body_limit:
                    gate.release(lease)
                    raise HTTPException(413, 'Video vượt giới hạn tải lên.')
            elif message['type'] == 'http.disconnect':
                gate.release(lease)
            return message

        async def bounded_send(message):
            try:
                await send(message)
            except BaseException:
                gate.release(lease)
                raise
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                gate.release(lease)

        try:
            if self.storage_check is not None:
                expected_bytes = min(int(length), body_limit) if length else body_limit
                try:
                    self.storage_check(expected_bytes)
                except HTTPException as exc:
                    return await JSONResponse({'detail': exc.detail}, exc.status_code,
                                              headers=exc.headers)(scope, receive, bounded_send)
            await self.app(scope, bounded_receive, bounded_send)
        finally:
            gate.release(lease)
