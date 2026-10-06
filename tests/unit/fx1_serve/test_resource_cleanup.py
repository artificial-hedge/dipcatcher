"""Regression checks for response and child-process lifecycle cleanup."""

from __future__ import annotations

import io
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import Mock, call

import pytest

from fx1.serve import backends, client, webhooks


class _TrackedBody(io.BytesIO):
    close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _http_error(status: int = 503) -> tuple[urllib.error.HTTPError, _TrackedBody]:
    body = _TrackedBody(b'{"error":"SYNTHETIC refusal"}')
    error = urllib.error.HTTPError(
        "https://provider.invalid/v1/chat/completions",
        status,
        "SYNTHETIC refusal",
        {},
        body,
    )
    return error, body


def test_backend_closes_http_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    error, body = _http_error()
    opener = Mock()
    opener.open.side_effect = error
    monkeypatch.setattr(backends.urllib.request, "build_opener", lambda *args: opener)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        backends._openai_chat_complete(
            "https://provider.invalid/v1/chat/completions",
            model="SYNTHETIC-model",
            messages=[{"role": "user", "content": "SYNTHETIC"}],
            timeout_s=1.0,
            api_key=None,
            label="SYNTHETIC",
        )

    assert body.close_calls == 1


def test_client_closes_http_error_after_read(monkeypatch: pytest.MonkeyPatch) -> None:
    error, body = _http_error(429)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(client.urllib.request, "urlopen", refuse)
    status, _headers, payload = client._urllib_transport(
        "GET", "https://harness.invalid/health", None, {}, 1.0
    )

    assert (status, payload) == (429, b'{"error":"SYNTHETIC refusal"}')
    assert body.close_calls == 1


def test_webhook_closes_each_http_error_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient 5xx closes its connection before the next attempt."""
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
    monkeypatch.setattr(time, "sleep", lambda _delay: None)

    class RefuseHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"SYNTHETIC refusal"}')

        def log_message(self, *_args: Any) -> None:
            pass

    created: list[webhooks._PinnedHTTPConnection] = []
    real_init = webhooks._PinnedHTTPConnection.__init__
    real_close = webhooks._PinnedHTTPConnection.close

    def tracked_init(self: webhooks._PinnedHTTPConnection, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self._close_calls = 0  # type: ignore[attr-defined]
        created.append(self)

    def counted_close(self: webhooks._PinnedHTTPConnection) -> None:
        self._close_calls += 1  # type: ignore[attr-defined]
        real_close(self)

    monkeypatch.setattr(webhooks._PinnedHTTPConnection, "__init__", tracked_init)
    monkeypatch.setattr(webhooks._PinnedHTTPConnection, "close", counted_close)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RefuseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ok, message, attempts = webhooks.deliver_signed(
            f"http://127.0.0.1:{server.server_address[1]}/hook",
            None,
            b'{"kind":"SYNTHETIC"}',
            max_attempts=2,
            backoff_s=0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert (ok, message, attempts) == (False, "callback endpoint returned 503", 2)
    assert len(created) == 2
    assert all(connection._close_calls >= 1 for connection in created)  # type: ignore[attr-defined]


def test_webhook_refuses_cross_origin_redirect_without_leaking_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
    target_hits: list[tuple[str, dict[str, str], bytes]] = []
    source_hits: list[tuple[str, dict[str, str], bytes]] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def _capture(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            target_hits.append((self.command, dict(self.headers), self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

        do_GET = _capture
        do_POST = _capture

        def log_message(self, *_args: Any) -> None:
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_url = f"http://127.0.0.1:{target.server_address[1]}/capture"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            source_hits.append((self.command, dict(self.headers), self.rfile.read(length)))
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True) for server in (source, target)
    ]
    for thread in threads:
        thread.start()
    try:
        ok, message, attempts = webhooks.deliver_signed(
            f"http://127.0.0.1:{source.server_address[1]}/redirect",
            "SYNTHETIC-webhook-secret",
            b'{"kind":"SYNTHETIC private record"}',
            max_attempts=1,
            backoff_s=0,
        )
    finally:
        for server in (source, target):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)

    assert (ok, message, attempts) == (False, "callback endpoint returned 302", 1)
    assert len(source_hits) == 1
    assert source_hits[0][0] == "POST"
    assert source_hits[0][2] == b'{"kind":"SYNTHETIC private record"}'
    assert target_hits == []


def test_local_backend_reaps_engine_after_forced_kill() -> None:
    proc = Mock()
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 5), 0]
    backend = object.__new__(backends.LocalFx1Backend)
    backend._engine_lock = threading.Lock()
    backend._proc = proc

    backend.close()

    assert backend._proc is None
    proc.terminate.assert_called_once_with()
    proc.kill.assert_called_once_with()
    assert proc.wait.call_args_list == [call(timeout=5), call()]
