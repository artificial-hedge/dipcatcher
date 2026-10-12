"""OpenAI-compatible loopback stub for hosted-model tests.

The helper is stdlib-only so either import works:

    from tests.fx1.stub_model_server import Reply, StubModelServer
    from .stub_model_server import StubModelServer  # inside tests.fx1

Bind is ``127.0.0.1:0`` (HTTP, not TLS). Use it as a context manager, or call
``start()`` / ``close()``. ``close()`` interrupts in-flight artificial delays
and does not wait them out.

Queued replies are served FIFO and skip the built-in authenticator. When the
queue is empty, ``set_default`` replaces the built-in policy; otherwise
:meth:`StubModelServer.builtin_reply` applies:

* model missing from ``keys`` -> HTTP 404 ``model_not_found``
* ``Authorization: Bearer <that model's key>`` missing or wrong -> HTTP 401
* else ``"stub <model> reply"`` with echoed ``model``, token usage, an
  ``x-request-id`` (``stub-N``), and ``system_fingerprint`` ``stub``

Requests are recorded *before* a queued delay, so ``wait_for_requests``
unblocks while the handler is still holding the response. The log stores a
fingerprint of the presented bearer token (last four characters), never the
token itself, and never the ``Authorization`` header. ``last_fault`` is the
exception type name from the last handler failure, or None.

``keys`` maps a model name to the bearer token that model accepts. This is a
test double: it does not prove anything about a live provider.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_TICK_S = 0.05
_HANDLER_JOIN_S = 1.0
_SERVE_JOIN_S = 2.0


def _fingerprint(secret: str) -> str:
    tail = secret[-4:] if len(secret) >= 4 else secret
    return f"...{tail}"


def _bearer(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, sep, rest = header.strip().partition(" ")
    if sep != " " or scheme.lower() != "bearer":
        return None
    token = rest.strip()
    if not token or " " in token:
        return None
    return token


def _same_secret(presented: str | None, expected: str | None) -> bool:
    if presented is None or expected is None or len(presented) != len(expected):
        return False
    return hmac.compare_digest(presented, expected)


def _openai_error(
    message: str, code: str, err_type: str = "invalid_request_error"
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "code": code}}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload)).encode("utf-8")


def _parse_json_object(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _content_length(handler: BaseHTTPRequestHandler) -> int | None:
    raw = handler.headers.get("Content-Length")
    if raw is None:
        return 0
    try:
        length = int(raw)
    except ValueError:
        return None
    if length < 0:
        return None
    return length


@dataclass(frozen=True)
class RecordedRequest:
    """One request the stub accepted. ``body`` is a shallow copy of the JSON object."""

    method: str
    path: str
    model: str | None
    n_messages: int
    has_tools: bool
    auth_ok: bool
    auth_fingerprint: str
    body: dict[str, Any] | None


@dataclass(frozen=True)
class Reply:
    """A scripted HTTP response. Prefer the factories; direct construction is for odd bodies."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    delay_s: float = 0.0
    discard: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(self, "body", bytes(self.body))

    @staticmethod
    def ok(
        text: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        usage: dict[str, int] | None = None,
        served_model: str | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        finish_reason: str | None = None,
        system_fingerprint: str | None = None,
    ) -> Reply:
        """HTTP 200 chat completion. ``served_model=None`` omits the ``model`` field."""
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        reason = finish_reason
        if reason is None:
            reason = "tool_calls" if tool_calls else "stop"
        payload: dict[str, Any] = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        }
        if served_model is not None:
            payload["model"] = served_model
        if usage is not None:
            payload["usage"] = usage
        if system_fingerprint is not None:
            payload["system_fingerprint"] = system_fingerprint
        return Reply(status=200, headers=dict(headers or {}), body=_json_bytes(payload))

    @staticmethod
    def error(
        status: int,
        headers: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Reply:
        """HTTP error. ``body`` defaults to an OpenAI-style ``{"error": ...}`` object."""
        payload = (
            dict(body) if body is not None else _openai_error("error", "error", "server_error")
        )
        return Reply(status=status, headers=dict(headers or {}), body=_json_bytes(payload))

    @staticmethod
    def delay(seconds: float, reply: Reply) -> Reply:
        """Hold ``reply`` for ``seconds`` after the request is recorded. ``close()`` interrupts it."""
        return Reply(
            status=reply.status,
            headers=reply.headers,
            body=reply.body,
            delay_s=max(0.0, float(seconds)),
            discard=reply.discard,
        )

    @staticmethod
    def malformed(body: bytes = b"<<<not-json>>>") -> Reply:
        """HTTP 200 whose body is not a chat completion."""
        return Reply(status=200, headers={"Content-Type": "application/json"}, body=body)

    @staticmethod
    def drop() -> Reply:
        """Close the socket without a status line."""
        return Reply(discard=True)

    @staticmethod
    def redirect(location: str, status: int = 302) -> Reply:
        """A redirect the client under test must not follow."""
        return Reply(status=status, headers={"Location": location}, body=b"")


class _StubHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    owner: StubModelServer

    def process_request(self, request: Any, client_address: Any) -> None:
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address),
            name="fx1-stub-handler",
            daemon=True,
        )
        self.owner.track_handler(thread)
        thread.start()


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _StubHTTPServer

    def do_POST(self) -> None:  # noqa: N802 — stdlib handler name
        self.server.owner.handle(self)

    def do_GET(self) -> None:  # noqa: N802
        self.server.owner.handle(self)

    def do_PUT(self) -> None:  # noqa: N802
        self.server.owner.handle(self)

    def do_DELETE(self) -> None:  # noqa: N802
        self.server.owner.handle(self)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — stdlib signature
        del format, args


class StubModelServer:
    """Loopback chat-completions stub. See the module docstring for the contract."""

    def __init__(self, keys: Mapping[str, str] | None = None) -> None:
        self._keys = dict(keys or {})
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._handler_lock = threading.Lock()
        self._handlers: list[threading.Thread] = []
        self._queue: list[Reply] = []
        self._requests: list[RecordedRequest] = []
        self._default: Callable[[RecordedRequest], Reply] | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._port = 0
        self._closed = False
        self.last_fault: str | None = None
        self._httpd: _StubHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> StubModelServer:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        """Bind ``127.0.0.1:0`` and serve in a daemon thread. Idempotent until :meth:`close`."""
        if self._closed:
            raise RuntimeError("StubModelServer is closed")
        if self._httpd is not None:
            return
        httpd = _StubHTTPServer(("127.0.0.1", 0), _StubHandler)
        httpd.owner = self
        self._httpd = httpd
        address = httpd.server_address
        self._port = int(address[1])
        thread = threading.Thread(target=self._serve, name="fx1-stub-model", daemon=True)
        self._thread = thread
        thread.start()

    def close(self) -> None:
        """Stop accepting, wake delayed handlers, and release the port. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        httpd = self._httpd
        thread = self._thread
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=_SERVE_JOIN_S)
        for handler in self._handler_threads():
            handler.join(timeout=_HANDLER_JOIN_S)
        self._httpd = None

    @property
    def port(self) -> int:
        self._require_running()
        return self._port

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:PORT`` with no path."""
        self._require_running()
        return f"http://127.0.0.1:{self._port}"

    @property
    def url(self) -> str:
        """Chat-completions URL, ``{base_url}/v1/chat/completions``."""
        return f"{self.base_url}/v1/chat/completions"

    @property
    def requests(self) -> list[RecordedRequest]:
        """Snapshot of the request log (oldest first)."""
        with self._lock:
            return list(self._requests)

    def queue(self, reply: Reply) -> None:
        """Append a scripted reply. Served in order, ahead of the default handler."""
        with self._lock:
            self._queue.append(reply)

    def set_default(self, handler: Callable[[RecordedRequest], Reply] | None) -> None:
        """Install the queue-empty handler. ``None`` restores :meth:`builtin_reply`."""
        with self._lock:
            self._default = handler

    def wait_for_requests(self, count: int = 1, timeout: float = 2.0) -> bool:
        """Block until ``count`` requests are recorded, or return False on timeout."""
        if count < 1:
            return True
        end = time.monotonic() + timeout
        with self._cv:
            while len(self._requests) < count:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True

    def builtin_reply(self, request: RecordedRequest) -> Reply:
        """Queue-empty policy: 404 unknown model, 401 bad key, else an echo completion."""
        model = request.model
        if model is None:
            return self._error_reply(400, "model is required", "missing_model")
        if model not in self._keys:
            return self._error_reply(404, f"The model {model!r} does not exist", "model_not_found")
        if not request.auth_ok:
            return self._error_reply(401, "Incorrect API key", "invalid_api_key")
        prompt = request.n_messages if request.n_messages > 0 else 1
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": 1,
            "total_tokens": prompt + 1,
        }
        return Reply.ok(
            f"stub {model} reply",
            usage=usage,
            served_model=model,
            headers={"x-request-id": self._next_id()},
            system_fingerprint="stub",
        )

    def track_handler(self, thread: threading.Thread) -> None:
        with self._handler_lock:
            self._handlers.append(thread)

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            self._handle(handler)
        except Exception as exc:
            self.last_fault = type(exc).__name__
            _send(
                handler,
                Reply.error(
                    500,
                    body=_openai_error("stub failed", "internal_error", "server_error"),
                ),
            )

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = _content_length(handler)
        if length is None:
            _send(handler, self._error_reply(400, "invalid Content-Length", "invalid_request"))
            return
        if length > _MAX_REQUEST_BYTES:
            _send(handler, self._error_reply(413, "request too large", "request_too_large"))
            return
        raw = handler.rfile.read(length) if length else b""
        record = self._record(
            handler.command or "", handler.path, handler.headers.get("Authorization"), raw
        )
        reply = self._next_reply(record)
        if not self._pause(reply.delay_s):
            _drop_connection(handler)
            return
        if reply.discard:
            _drop_connection(handler)
            return
        _send(handler, reply)

    def _serve(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        httpd.serve_forever(poll_interval=0.1)

    def _require_running(self) -> None:
        if self._port == 0 or self._httpd is None:
            raise RuntimeError(
                "StubModelServer is not running; enter the context manager or call start()"
            )

    def _record(
        self, method: str, path: str, authorization: str | None, raw: bytes
    ) -> RecordedRequest:
        parsed = _parse_json_object(raw)
        model = _model_of(parsed)
        messages = parsed.get("messages") if parsed is not None else None
        n_messages = len(messages) if isinstance(messages, list) else 0
        tools = parsed.get("tools") if parsed is not None else None
        has_tools = isinstance(tools, list) and len(tools) > 0
        token = _bearer(authorization)
        expected = self._keys.get(model) if model is not None else None
        record = RecordedRequest(
            method=method,
            path=path,
            model=model,
            n_messages=n_messages,
            has_tools=has_tools,
            auth_ok=_same_secret(token, expected),
            auth_fingerprint=_fingerprint(token) if token else "",
            body=dict(parsed) if parsed is not None else None,
        )
        with self._cv:
            self._requests.append(record)
            self._cv.notify_all()
        return record

    def _next_reply(self, record: RecordedRequest) -> Reply:
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            handler = self._default
        if handler is None:
            return self.builtin_reply(record)
        try:
            reply = handler(record)
        except Exception as exc:
            self.last_fault = type(exc).__name__
            return self._error_reply(500, "stub handler failed", "internal_error")
        if not isinstance(reply, Reply):
            return self._error_reply(500, "stub handler failed", "internal_error")
        return reply

    def _pause(self, delay_s: float) -> bool:
        """Return True when the reply should be sent. False means the server is stopping."""
        if self._stop.is_set():
            return False
        if delay_s <= 0:
            return True
        end = time.monotonic() + delay_s
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return not self._stop.is_set()
            if self._stop.wait(min(_TICK_S, remaining)):
                return False

    def _error_reply(self, status: int, message: str, code: str) -> Reply:
        err_type = "server_error" if status >= 500 else "invalid_request_error"
        return Reply.error(
            status,
            headers={"x-request-id": self._next_id()},
            body=_openai_error(message, code, err_type),
        )

    def _next_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"stub-{self._seq}"

    def _handler_threads(self) -> list[threading.Thread]:
        with self._handler_lock:
            return list(self._handlers)


def _model_of(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    model = parsed.get("model")
    if isinstance(model, str) and model:
        return model
    return None


def _drop_connection(handler: BaseHTTPRequestHandler) -> None:
    handler.close_connection = True
    with contextlib.suppress(OSError):
        handler.connection.shutdown(socket.SHUT_RDWR)


def _send(handler: BaseHTTPRequestHandler, reply: Reply) -> None:
    try:
        handler.close_connection = True
        handler.send_response(reply.status)
        _write_headers(handler, reply)
        handler.end_headers()
        if reply.body:
            handler.wfile.write(reply.body)
    except OSError:
        handler.close_connection = True


def _write_headers(handler: BaseHTTPRequestHandler, reply: Reply) -> None:
    skip = {"content-length", "connection", "transfer-encoding"}
    saw_type = False
    for key, value in reply.headers.items():
        if key.lower() in skip:
            continue
        if key.lower() == "content-type":
            saw_type = True
        handler.send_header(key, value)
    if not saw_type:
        handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(reply.body)))
    handler.send_header("Connection", "close")


__all__ = ["RecordedRequest", "Reply", "StubModelServer"]
