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
        self.inflight = False
        self.lock = threading.Lock()

    def admit(self, busy):
        with self.lock:
            now = time.monotonic()
            while self.starts and now - self.starts[0] >= self.window_seconds:
                self.starts.popleft()
            if self.inflight or busy():
                return 409, BUSY_MESSAGE
            if len(self.starts) >= self.attempts:
                return 429, "Có quá nhiều lượt tải video. Vui lòng thử lại sau một phút."
            self.starts.append(now)
            self.inflight = True
            return None

    def release(self):
        with self.lock:
            self.inflight = False


class PublicUploadMiddleware:
    def __init__(self, app, public_mode, enabled, max_bytes, busy, gate, idle_timeout=60, total_timeout=600):
        self.app = app
        self.public_mode = public_mode
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.busy = busy
        self.gate = gate
        self.idle_timeout = idle_timeout
        self.total_timeout = total_timeout

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
        rejected = gate.admit(self.busy)
        if rejected:
            code, detail = rejected
            return await JSONResponse({'detail': detail}, code, headers={'Retry-After': '60' if code == 429 else '5'})(scope, receive, send)
        consumed = 0
        deadline = time.monotonic() + self.total_timeout

        async def bounded_receive():
            nonlocal consumed
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(408, 'Tải video quá thời gian cho phép. Vui lòng thử lại.')
            try:
                message = await asyncio.wait_for(receive(), timeout=min(self.idle_timeout, remaining))
            except asyncio.TimeoutError:
                raise HTTPException(408, 'Kết nối tải video bị gián đoạn. Vui lòng thử lại.') from None
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > body_limit:
                    raise HTTPException(413, 'Video vượt giới hạn tải lên.')
            return message

        try:
            await self.app(scope, bounded_receive, send)
        finally:
            gate.release()
