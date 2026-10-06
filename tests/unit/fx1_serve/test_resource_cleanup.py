"""Regression checks for response and child-process lifecycle cleanup."""

from __future__ import annotations

import io
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
    errors = [_http_error()[0], _http_error()[0]]
    bodies = [error.fp for error in errors]

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise errors.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    ok, message, attempts = webhooks.deliver_signed(
        "https://callback.invalid/hook",
        None,
        b'{"kind":"SYNTHETIC"}',
        max_attempts=2,
        backoff_s=0,
    )

    assert (ok, message, attempts) == (False, "callback endpoint returned 503", 2)
    assert all(isinstance(body, _TrackedBody) and body.close_calls == 1 for body in bodies)


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
