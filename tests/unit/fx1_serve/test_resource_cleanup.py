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
    """Each 5xx attempt closes its response and connection before the next."""
    closed: list[object] = []

    class _RefusalResponse:
        status = 503

        def __enter__(self) -> _RefusalResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            closed.append(self)

    class _RefusalConnection:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> _RefusalResponse:
            return _RefusalResponse()

        def close(self) -> None:
            closed.append(self)

    # Delivery dials a validated numeric address through the pinned
    # connection class — patch both, not urllib's opener (a dead seam).
    monkeypatch.setattr(webhooks, "_resolved_addresses", lambda _host, _port: ("93.184.216.34",))
    monkeypatch.setattr(webhooks, "_PinnedHTTPSConnection", _RefusalConnection)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    ok, message, attempts = webhooks.deliver_signed(
        "https://callback.invalid/hook",
        None,
        b'{"kind":"SYNTHETIC"}',
        max_attempts=2,
        backoff_s=0,
    )

    assert (ok, message, attempts) == (False, "callback endpoint returned 503", 2)
    assert [type(o) for o in closed] == [
        _RefusalResponse,
        _RefusalConnection,
        _RefusalResponse,
        _RefusalConnection,
    ]


def test_webhook_refuses_cross_origin_redirect_without_leaking_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the SSRF guard is satisfied: loopback sinks opt in via the env flag
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
