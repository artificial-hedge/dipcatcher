"""Live and adversarial checks for endpoint URLs, credentials, and the model client.

Loopback sockets and the stub server stand in for a provider. Secrets in
these tests are fixtures; assertions keep them out of errors, logs, and
child-process environments.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from email.utils import formatdate
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive import profiles
from fx1.interactive.endpoint import EndpointError, normalize_endpoint
from fx1.interactive.model_client import (
    ErrorKind,
    ModelClient,
    ModelError,
    RetryPolicy,
    retry_backoff_delay,
)
from fx1.interactive.redact import DEFAULT_REGISTRY
from fx1.serve.api import create_app
from fx1.serve.backends import MOONSHOT_API_URL, ToolCompletion
from tests.fx1.stub_model_server import Reply, StubModelServer

KEY = "sk-r2-live-key-AAAA"
LITE = "sk-r2-lite-key-BBBB"
TOKEN = "evil-token"
_PROXY_NAMES = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)
_CHILD = """
import sys
from pathlib import Path
from fx1.interactive import profiles
config, model, key_path, url = sys.argv[1:5]
profiles.set_endpoint(model, Path(key_path).read_text(encoding="utf-8"), url)
"""


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))
    for name in (
        profiles.ENV_API_KEY,
        profiles.ENV_BASE_URL,
        profiles.ENV_FX1_API_KEY,
        profiles.ENV_FX1_BASE_URL,
        profiles.ENV_FX1_LITE_API_KEY,
        profiles.ENV_FX1_LITE_BASE_URL,
        *_PROXY_NAMES,
    ):
        monkeypatch.delenv(name, raising=False)
    DEFAULT_REGISTRY.clear()
    yield
    DEFAULT_REGISTRY.clear()


def _msg(text: str = "hello") -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _client(url: str, **kwargs: Any) -> ModelClient:
    params: dict[str, Any] = {
        "api_key": KEY,
        "api_url": url,
        "timeout_s": 2.0,
        "retry": RetryPolicy(max_attempts=2, base_delay_s=0.05, max_delay_s=0.2, jitter=0),
    }
    if not kwargs.pop("env_proxy", False):
        params["proxies"] = lambda: {}
    params.update(kwargs)
    return ModelClient(**params)


def _clean(text: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in text


def _error_body(message: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


class _Accepts:
    """Counts TCP accepts and drops them. Used to prove a call never connects."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = int(self.sock.getsockname()[1])
        self.accepted = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fx1-accepts", daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def close(self) -> None:
        self._stop.set()
        self.sock.close()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        self.sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _addr = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.accepted += 1
            conn.close()


class _ProxyLog:
    """Records bytes from one client. CONNECT is answered so TLS can start."""

    def __init__(self) -> None:
        self.blob = b""
        self.connected = threading.Event()
        self.recorded = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._run, name="fx1-proxy-log", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        self._sock.settimeout(3)
        try:
            conn, _addr = self._sock.accept()
        except OSError:
            self.recorded.set()
            return
        self.connected.set()
        try:
            self.blob = _slurp_proxy(conn)
        finally:
            conn.close()
            self.recorded.set()


def _slurp_proxy(conn: socket.socket) -> bytes:
    conn.settimeout(1.0)
    data = b""
    try:
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = conn.recv(4096)
            if not chunk:
                return data
            data += chunk
        if data.startswith(b"CONNECT"):
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            conn.settimeout(0.5)
            while len(data) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return data
    return data


def _read_headers(conn: socket.socket) -> bytes:
    conn.settimeout(2)
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def _once(handler: Callable[[socket.socket], None]) -> tuple[int, threading.Thread, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = int(sock.getsockname()[1])

    def run() -> None:
        sock.settimeout(3)
        try:
            conn, _addr = sock.accept()
        except OSError:
            return
        try:
            handler(conn)
        finally:
            conn.close()

    thread = threading.Thread(target=run, name="fx1-once", daemon=True)
    thread.start()
    return port, thread, sock


def _child_env(config: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", "src"),
        "FX1_CONFIG_DIR": str(config),
        "FX1_SESSION_DIR": str(config / "sessions"),
    }
    lang = os.environ.get("LANG")
    if lang:
        env["LANG"] = lang
    return env


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        f"http://[::1%{TOKEN}]/v1",
        f"http://[::1%25{TOKEN}]/v1",
        f"http://127.0.0.1%00.{TOKEN}/v1",
        f"https://h.example/v1/%2e%2e/{TOKEN}",
        f"https://h.example/v1%2f{TOKEN}",
        f"https://h.example/v1%5c{TOKEN}",
        f"https://h.example/v1/../{TOKEN}",
        f"https://h.example/v1\\..\\{TOKEN}",
        f"https://{TOKEN}.example./v1",
        f"https://user:{TOKEN}@h.example/v1",
        f"https://h.example/v1\r\nX-Injected: {TOKEN}",
        f"https://h.example/v1\n{TOKEN}",
        f"https://ex\u0430mple.{TOKEN}/v1",
        f"https://h.example/{TOKEN}" + ("a" * 10_000),
        f"http://localhost/{TOKEN}",
        f"http://0.0.0.0/{TOKEN}",
        f"http://2130706433/{TOKEN}",
        "https://h.example:0/v1",
        "https://h.example:65536/v1",
    ],
)
def test_endpoint_rejects_attacks_without_echoing_the_url(raw: str) -> None:
    with pytest.raises(EndpointError) as ei:
        normalize_endpoint(raw)
    text = str(ei.value)
    _clean(text, TOKEN, raw)
    assert len(text) < 200


def test_literal_loopback_http_is_distinct_from_localhost() -> None:
    with pytest.raises(EndpointError) as ei:
        normalize_endpoint(f"http://localhost/{TOKEN}")
    _clean(str(ei.value), TOKEN)
    spec = normalize_endpoint("http://127.0.0.1:9/v1")
    assert spec.host == "127.0.0.1"
    assert spec.completion_url == "http://127.0.0.1:9/v1/chat/completions"
    v6 = normalize_endpoint("http://[::1]:9/v1")
    assert v6.host == "::1"
    mapped = normalize_endpoint("http://[::ffff:127.0.0.1]/v1")
    assert mapped.host == "::ffff:127.0.0.1"
    edge = normalize_endpoint("https://h.example:1/v1")
    assert edge.origin == "https://h.example:1"
    top = normalize_endpoint("https://h.example:65535/v1")
    assert top.origin == "https://h.example:65535"
    assert "artificialhedge" not in spec.completion_url


def test_client_rejects_smuggled_loopback_before_connect() -> None:
    gate = _Accepts()
    try:
        url = f"http://127.0.0.1%00.{TOKEN}:{gate.port}/v1/chat/completions"
        with pytest.raises(RuntimeError) as ei:
            _client(url, timeout_s=0.5, retry=RetryPolicy(max_attempts=1)).complete(_msg())
        _clean(str(ei.value), TOKEN, url)
        time.sleep(0.05)
        assert gate.accepted == 0
    finally:
        gate.close()


def test_client_rejects_zone_and_dot_segments_before_connect() -> None:
    gate = _Accepts()
    try:
        zoned = f"http://[::1%{TOKEN}]:{gate.port}/v1"
        dotted = f"http://127.0.0.1:{gate.port}/v1/../{TOKEN}"
        for url in (zoned, dotted):
            with pytest.raises(RuntimeError) as ei:
                _client(url, timeout_s=0.5, retry=RetryPolicy(max_attempts=1)).complete(_msg())
            _clean(str(ei.value), TOKEN, url)
        time.sleep(0.05)
        assert gate.accepted == 0
    finally:
        gate.close()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        f"sk-r2-{TOKEN}\r\nX-Injected: yes",
        f"sk-r2-{TOKEN}\nmore",
        f"sk-r2 {TOKEN}",
        f"sk-r2-{TOKEN}\x00tail",
        f"sk-модель-{TOKEN}",
    ],
)
def test_unsafe_key_is_rejected_before_any_socket(
    key: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    gate = _Accepts()
    try:
        with pytest.raises(ValueError) as ei:
            ModelClient(
                api_key=key,
                api_url=gate.url(),
                timeout_s=0.5,
                retry=RetryPolicy(max_attempts=1),
                proxies=lambda: {},
            ).complete(_msg())
        rendered = " ".join((str(ei.value), repr(ei.value), traceback.format_exc(), caplog.text))
        _clean(rendered, key, TOKEN)
        time.sleep(0.05)
        assert gate.accepted == 0
    finally:
        gate.close()


def test_success_and_error_logs_omit_the_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with StubModelServer(keys={"fx1": KEY}) as srv:
        client = _client(srv.url, retry=RetryPolicy(max_attempts=1))
        assert KEY not in repr(client)
        client.complete(_msg())
        srv.queue(Reply.error(500, body=_error_body(f"boom {KEY}", "server_error")))
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        rendered = " ".join((str(ei.value), repr(ei.value), ei.value.summary, caplog.text))
        _clean(rendered, KEY)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_status_does_not_forward_authorization(status: int) -> None:
    with (
        StubModelServer(keys={"fx1": KEY}) as origin,
        StubModelServer(keys={"fx1": KEY}) as sink,
    ):
        origin.queue(Reply.redirect(f"{sink.url}?{TOKEN}=1", status=status))
        client = _client(origin.url, retry=RetryPolicy(max_attempts=2, jitter=0))
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == ErrorKind.ROUTE_NOT_FOUND
        assert ei.value.retryable is False
        assert sink.requests == []
        assert len(origin.requests) == 1
        _clean(str(ei.value), KEY, TOKEN)


def test_ambient_http_proxy_does_not_receive_the_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _ProxyLog()
    try:
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{proxy.port}")
        with StubModelServer(keys={"fx1": KEY}) as srv:
            client = _client(srv.url, env_proxy=True, retry=RetryPolicy(max_attempts=1))
            assert client.complete(_msg()) == "stub fx1 reply"
            assert srv.requests[0].auth_ok is True
        assert proxy.connected.is_set() is False
        _clean(proxy.blob.decode("latin-1", errors="replace"), KEY)
    finally:
        proxy.close()


def test_https_proxy_preamble_has_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy = _ProxyLog()
    try:
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{proxy.port}")
        monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy.port}")
        client = ModelClient(
            api_key=KEY,
            api_url="https://example.com/v1/chat/completions",
            timeout_s=2.0,
            retry=RetryPolicy(max_attempts=1),
        )
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert proxy.recorded.wait(2)
        blob = proxy.blob.decode("latin-1", errors="replace")
        _clean(blob, KEY)
        assert "Authorization" not in blob
        _clean(str(ei.value), KEY)
    finally:
        proxy.close()


# ---------------------------------------------------------------------------
# Failure matrix, retries, cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "kind", "retryable"),
    [
        (401, "invalid_api_key", ErrorKind.AUTH, False),
        (403, "permission", ErrorKind.PERMISSION, False),
        (404, "model_not_found", ErrorKind.UNKNOWN_MODEL, False),
        (429, "rate_limit", ErrorKind.RATE_LIMITED, True),
        (500, "server_error", ErrorKind.SERVER, True),
        (502, "bad_gateway", ErrorKind.SERVER, True),
        (503, "unavailable", ErrorKind.SERVER, True),
        (504, "gateway_timeout", ErrorKind.SERVER, True),
    ],
)
def test_status_matrix(status: int, code: str, kind: ErrorKind, retryable: bool) -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.error(status, body=_error_body(f"nope {KEY}", code)))
        client = _client(srv.url, retry=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == kind
        assert ei.value.retryable is retryable
        assert ei.value.status == status
        _clean(str(ei.value), KEY)
        assert len(srv.requests) == 1


def test_retry_after_shapes() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(
            Reply.error(429, headers={"Retry-After": "0.2"}, body=_error_body("slow", "rate_limit"))
        )
        client = _client(
            srv.url,
            retry=RetryPolicy(
                max_attempts=2, base_delay_s=5, jitter=0, honor_retry_after_up_to_s=30
            ),
        )
        started = time.monotonic()
        assert client.complete(_msg()) == "stub fx1 reply"
        assert 0.15 <= time.monotonic() - started < 1.5
        assert len(srv.requests) == 2

        anchor = float(int(time.time()) + 10)
        when = formatdate(timeval=anchor + 2, usegmt=True)
        srv.queue(Reply.error(429, headers={"Retry-After": when}, body=_error_body("slow", "rate")))
        dated_sleeps: list[float] = []
        dated = _client(
            srv.url,
            now=lambda: anchor,
            sleep=dated_sleeps.append,
            retry=RetryPolicy(max_attempts=2, base_delay_s=5, jitter=0),
        )
        assert dated.complete(_msg()) == "stub fx1 reply"
        assert dated_sleeps == [pytest.approx(2.0)]

        slept: list[float] = []

        def sleep(seconds: float) -> None:
            if seconds > 1:
                raise AssertionError(seconds)
            slept.append(seconds)

        for header in ("not-a-date", "-4", "1000000000"):
            slept.clear()
            srv.queue(
                Reply.error(429, headers={"Retry-After": header}, body=_error_body("slow", "rate"))
            )
            if header == "1000000000":
                bounded = _client(
                    srv.url,
                    sleep=sleep,
                    retry=RetryPolicy(max_attempts=3, base_delay_s=0.05, jitter=0),
                )
                with pytest.raises(ModelError) as ei:
                    bounded.complete(_msg())
                assert ei.value.kind == ErrorKind.RATE_LIMITED
                assert slept == []
            else:
                fallback = _client(
                    srv.url,
                    sleep=sleep,
                    retry=RetryPolicy(max_attempts=2, base_delay_s=0.05, jitter=0),
                )
                assert fallback.complete(_msg()) == "stub fx1 reply"
                assert slept == [pytest.approx(0.05)]


def test_retries_never_exceed_policy_and_jitter_is_bounded() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0, jitter=0.25)
    for attempt in range(1, 8):
        capped = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
        low = capped * (1 - policy.jitter)
        high = capped * (1 + policy.jitter)
        for unit in (0.0, 0.25, 0.5, 0.75, 1.0):
            delay = retry_backoff_delay(policy, attempt, unit)
            assert low - 1e-9 <= delay <= high + 1e-9
    with StubModelServer(keys={"fx1": KEY}) as srv:
        for _ in range(5):
            srv.queue(Reply.error(503, body=_error_body("down", "unavailable")))
        client = _client(
            srv.url,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05, jitter=0),
        )
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == ErrorKind.SERVER
        assert len(srv.requests) == 3


def test_slowloris_timeout_and_deadline_hold() -> None:
    def stall(conn: socket.socket) -> None:
        _read_headers(conn)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000000\r\n\r\n"
        )
        time.sleep(3)

    port, thread, sock = _once(stall)
    try:
        started = time.monotonic()
        with pytest.raises(ModelError) as ei:
            _client(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                timeout_s=0.3,
                retry=RetryPolicy(max_attempts=1),
            ).complete(_msg())
        assert ei.value.kind == ErrorKind.TIMEOUT
        assert time.monotonic() - started < 1.2
        _clean(str(ei.value), KEY)
    finally:
        sock.close()
        thread.join(timeout=1)

    port, thread, sock = _once(stall)
    try:
        started = time.monotonic()
        with pytest.raises(ModelError) as ei:
            _client(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                timeout_s=2.0,
                retry=RetryPolicy(max_attempts=3, jitter=0),
            ).complete(_msg(), deadline=started + 0.3)
        assert ei.value.kind == ErrorKind.DEADLINE
        assert time.monotonic() - started < 1.0
        _clean(str(ei.value), KEY)
    finally:
        sock.close()
        thread.join(timeout=1)


def test_reset_mid_body_fails_closed() -> None:
    def rst(conn: socket.socket) -> None:
        _read_headers(conn)
        conn.sendall(
            b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 80\r\n\r\n{"cho'
        )
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()

    port, thread, sock = _once(rst)
    try:
        started = time.monotonic()
        with pytest.raises(ModelError) as ei:
            _client(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                timeout_s=1.0,
                retry=RetryPolicy(max_attempts=1),
            ).complete(_msg())
        assert ei.value.kind in {ErrorKind.NETWORK, ErrorKind.MALFORMED}
        assert time.monotonic() - started < 2
        _clean(str(ei.value), KEY)
    finally:
        sock.close()
        thread.join(timeout=1)


def test_truncated_html_and_wrong_type_are_not_completions() -> None:
    body = json.dumps(
        {
            "model": "fx1",
            "choices": [{"message": {"content": TOKEN}, "finish_reason": "stop"}],
        }
    ).encode()
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.malformed(b'{"choices":[{"message":{"content":"hi"'))
        client = _client(srv.url, retry=RetryPolicy(max_attempts=3, base_delay_s=0.05, jitter=0))
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == ErrorKind.MALFORMED
        assert len(srv.requests) == 1

        srv.queue(
            Reply(
                status=200,
                headers={"Content-Type": "text/html"},
                body=f"<html>{TOKEN}</html>".encode(),
            )
        )
        with pytest.raises(ModelError) as html:
            client.complete(_msg())
        assert html.value.kind == ErrorKind.MALFORMED
        assert "content-type" in html.value.summary
        _clean(str(html.value), TOKEN, KEY)

        srv.queue(
            Reply(
                status=200,
                headers={"Content-Type": "text/html", "Content-Encoding": "identity"},
                body=body,
            )
        )
        with pytest.raises(ModelError) as typed:
            client.complete(_msg())
        assert typed.value.kind == ErrorKind.MALFORMED
        assert TOKEN not in str(typed.value)

        srv.queue(
            Reply(
                status=200,
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
                body=body,
            )
        )
        with pytest.raises(ModelError) as gzipped:
            client.complete(_msg())
        assert gzipped.value.kind == ErrorKind.MALFORMED
        assert "content-encoding" in gzipped.value.summary
        _clean(str(gzipped.value), TOKEN, KEY)


def test_fifty_megabyte_body_is_capped() -> None:
    sent = {"n": 0}

    def bulk(conn: socket.socket) -> None:
        _read_headers(conn)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 52428800\r\n\r\n"
        )
        chunk = b"A" * 65536
        while sent["n"] < 50 * 1024 * 1024:
            try:
                conn.sendall(chunk)
            except OSError:
                return
            sent["n"] += len(chunk)

    port, thread, sock = _once(bulk)
    try:
        started = time.monotonic()
        with pytest.raises(ModelError) as ei:
            _client(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                timeout_s=2.0,
                max_response_bytes=65536,
                retry=RetryPolicy(max_attempts=1),
            ).complete(_msg())
        assert ei.value.kind == ErrorKind.MALFORMED
        assert "exceeded" in ei.value.summary
        assert sent["n"] < 2_000_000
        assert time.monotonic() - started < 3
        _clean(str(ei.value), KEY)
    finally:
        sock.close()
        thread.join(timeout=1)


def test_cancel_is_prompt_and_fifty_calls_do_not_leak_threads() -> None:
    def live() -> int:
        return sum(
            1 for item in threading.enumerate() if item.name == "fx1-model-http" and item.is_alive()
        )

    with StubModelServer(keys={"fx1": KEY}) as srv:
        before = live()
        client = _client(srv.url, timeout_s=2.0, retry=RetryPolicy(max_attempts=3, jitter=0))
        for _ in range(50):
            srv.queue(Reply.delay(30, Reply.ok("no", served_model="fx1")))
            cancel = threading.Event()
            caught: list[BaseException] = []

            def run(cancel: threading.Event = cancel, caught: list[BaseException] = caught) -> None:
                try:
                    client.complete(_msg(), cancel_event=cancel)
                except BaseException as exc:  # noqa: BLE001 — the test records the outcome
                    caught.append(exc)

            worker = threading.Thread(target=run)
            seen = len(srv.requests)
            worker.start()
            assert srv.wait_for_requests(seen + 1, timeout=2)
            cancel.set()
            worker.join(1.0)
            assert not worker.is_alive()
            assert len(caught) == 1
            assert isinstance(caught[0], ModelError)
            assert caught[0].kind == ErrorKind.CANCELLED
        assert len(srv.requests) == 50
        time.sleep(0.2)
        assert live() <= before


# ---------------------------------------------------------------------------
# Identity and probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("served", "matched"),
    [
        ("fx1", True),
        ("fx1-20261001", True),
        ("fx1-lite", False),
        ("fx1-lite-v2", False),
        ("fx1-lite-20261001", False),
        ("fx1-v2", False),
    ],
)
def test_identity_alias_is_a_date_suffix_not_a_prefix(served: str, matched: bool) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.ok("text", served_model=served, headers={"x-request-id": "rid"}))
        client = _client(
            srv.url,
            retry=RetryPolicy(max_attempts=1),
            emit=lambda name, fields: events.append((name, dict(fields))),
        )
        assert client.complete(_msg()) == "text"
        identity = client.last_identity
        assert identity is not None
        assert identity.requested == "fx1"
        assert identity.served == served
        assert identity.matched is matched
        assert srv.requests[0].body is not None
        assert srv.requests[0].body["model"] == "fx1"
        mismatches = [fields for name, fields in events if name == "model.identity_mismatch"]
        assert bool(mismatches) is (matched is False)


class _HarnessEcho:
    """In-process stand-in the harness calls. Text is a single plain word."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del messages, sampling
        return "pong"

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        del messages, sampling, tool_choice, parallel_tool_calls, logprobs, top_logprobs
        if not tools:
            return ToolCompletion(content="pong", tool_calls=None, finish_reason="stop")
        call = {"id": "call-1", "type": "function", "function": {"name": "ping", "arguments": "{}"}}
        return ToolCompletion(content=None, tool_calls=(call,), finish_reason="tool_calls")


def _serve_harness() -> tuple[Any, threading.Thread, str]:
    import logging

    import uvicorn

    logging.getLogger("fx1-api").setLevel(logging.ERROR)
    app = create_app(backend_resolver=lambda *_a, **_k: _HarnessEcho(), sse_keepalive_s=30.0)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="fx1-harness-serve", daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("harness serve did not bind")
        time.sleep(0.02)
    sock = server.servers[0].sockets[0]
    port = int(sock.getsockname()[1])
    return server, thread, f"http://127.0.0.1:{port}/v1/chat/completions"


def test_model_client_against_harness_serve() -> None:
    server, thread, url = _serve_harness()
    try:
        client = _client(url, timeout_s=3.0, retry=RetryPolicy(max_attempts=1))
        assert client.complete(_msg("hi")) == "pong"
        identity = client.last_identity
        assert identity is not None
        assert identity.requested == "fx1"
        assert identity.served == "hosted_k3"
        assert identity.matched is False
        assert identity.response_id is not None
        assert identity.response_id.startswith("chatcmpl-")
        assert identity.request_id
        tools = [
            {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
        ]
        completion = client.complete_with_tools(_msg("hi"), tools=tools)
        assert completion.finish_reason == "tool_calls"
        assert completion.tool_calls is not None
        assert completion.tool_calls[0]["function"]["name"] == "ping"
        probed = client.probe()
        assert probed.requested == "fx1"
        assert probed.response_id is not None
        assert probed.response_id.startswith("chatcmpl-")
        _clean(repr(client), KEY)
    finally:
        server.should_exit = True
        thread.join(timeout=2)


def test_alias_catalog_matches_profile_models() -> None:
    from fx1.interactive.model_client import _KNOWN_MODELS

    assert _KNOWN_MODELS == profiles.MODELS


def test_lite_date_suffix_matches_lite_only() -> None:
    with StubModelServer(keys={"fx1-lite": LITE}) as srv:
        srv.queue(Reply.ok("text", served_model="fx1-lite-20261001"))
        client = _client(srv.url, api_key=LITE, model="fx1-lite", retry=RetryPolicy(max_attempts=1))
        client.complete(_msg())
        identity = client.last_identity
        assert identity is not None
        assert identity.matched is True
        assert identity.requested == "fx1-lite"
        srv.queue(Reply.ok("again", served_model="fx1"))
        client.complete(_msg())
        assert client.last_identity is not None
        assert client.last_identity.matched is False
        assert [item.body["model"] for item in srv.requests if item.body] == [
            "fx1-lite",
            "fx1-lite",
        ]


def test_response_id_is_bounded_and_error_carries_request_id() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.ok("hi", served_model="fx1", headers={"x-request-id": "req-1"}))
        client = _client(srv.url, retry=RetryPolicy(max_attempts=1))
        assert client.complete(_msg()) == "hi"
        identity = client.last_identity
        assert identity is not None
        assert identity.response_id == "chatcmpl-stub"
        assert identity.request_id == "req-1"
        assert KEY not in repr(identity)

        planted = {
            "id": KEY,
            "model": "fx1",
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        }
        srv.queue(Reply(status=200, body=json.dumps(planted).encode()))
        client.complete(_msg())
        hidden = client.last_identity
        assert hidden is not None
        assert hidden.response_id is None
        assert KEY not in repr(hidden)

        srv.queue(
            Reply.error(
                401,
                headers={"x-request-id": f"err {KEY}"},
                body=_error_body("nope", "invalid_api_key"),
            )
        )
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.request_id is None
        _clean(str(ei.value), KEY)

        srv.queue(
            Reply.error(
                500,
                headers={"x-request-id": "err-ok"},
                body=_error_body("down", "server_error"),
            )
        )
        with pytest.raises(ModelError) as server:
            client.complete(_msg())
        assert server.value.request_id == "err-ok"
        assert server.value.kind == ErrorKind.SERVER


def test_unreported_identity_stays_unknown() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.ok("plain"))
        client = _client(srv.url, retry=RetryPolicy(max_attempts=1))
        client.complete(_msg())
        identity = client.last_identity
        assert identity is not None
        assert identity.served is None
        assert identity.matched is None


def test_probe_tolerates_missing_models_route_and_stops_on_auth() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.error(404, body=_error_body("missing", "not_found")))
        client = _client(srv.url, retry=RetryPolicy(max_attempts=1))
        identity = client.probe()
        assert identity.matched is True
        assert [item.method for item in srv.requests] == ["GET", "POST"]
        assert srv.requests[1].body is not None
        assert srv.requests[1].body["model"] == "fx1"

        srv.queue(Reply.error(405, body=_error_body("nope", "method_not_allowed")))
        assert client.probe().requested == "fx1"

        srv.queue(Reply.error(401, body=_error_body(f"bad {KEY}", "invalid_api_key")))
        before = len(srv.requests)
        with pytest.raises(ModelError) as ei:
            client.probe()
        assert ei.value.kind == ErrorKind.AUTH
        assert [item.method for item in srv.requests[before:]] == ["GET"]
        _clean(str(ei.value), KEY)


def test_probe_retries_completion_rate_limit_within_bound() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.error(404, body=_error_body("missing", "not_found")))
        srv.queue(
            Reply.error(
                429, headers={"Retry-After": "0.15"}, body=_error_body("slow", "rate_limit")
            )
        )
        client = _client(
            srv.url,
            retry=RetryPolicy(
                max_attempts=3, base_delay_s=5, jitter=0, honor_retry_after_up_to_s=30
            ),
        )
        started = time.monotonic()
        identity = client.probe()
        elapsed = time.monotonic() - started
        assert identity.matched is True
        assert [item.method for item in srv.requests] == ["GET", "POST", "POST"]
        assert 0.1 <= elapsed < 2
        posts = [item for item in srv.requests if item.method == "POST" and item.body]
        assert [item.body["model"] for item in posts] == ["fx1", "fx1"]


def test_probe_cancel_during_models_get_does_not_complete() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.delay(30, Reply.error(404, body=_error_body("missing", "not_found"))))
        client = _client(srv.url, timeout_s=2.0, retry=RetryPolicy(max_attempts=2, jitter=0))
        cancel = threading.Event()
        caught: list[BaseException] = []

        def run() -> None:
            try:
                client.probe(cancel_event=cancel)
            except BaseException as exc:  # noqa: BLE001 — the test records the outcome
                caught.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        assert srv.wait_for_requests(1, timeout=2)
        started = time.monotonic()
        cancel.set()
        worker.join(1.0)
        assert not worker.is_alive()
        assert time.monotonic() - started < 1.0
        assert len(caught) == 1
        assert isinstance(caught[0], ModelError)
        assert caught[0].kind == ErrorKind.CANCELLED
        assert [item.method for item in srv.requests] == ["GET"]


def test_probe_deadline_aborts_models_get() -> None:
    with StubModelServer(keys={"fx1": KEY}) as srv:
        srv.queue(Reply.delay(30, Reply.error(404, body=_error_body("missing", "not_found"))))
        client = _client(srv.url, timeout_s=2.0, retry=RetryPolicy(max_attempts=2, jitter=0))
        started = time.monotonic()
        with pytest.raises(ModelError) as ei:
            client.probe(deadline=started + 0.3)
        assert ei.value.kind == ErrorKind.DEADLINE
        assert time.monotonic() - started < 1.0
        assert [item.method for item in srv.requests] == ["GET"]


def test_probe_cancel_does_not_connect() -> None:
    gate = _Accepts()
    try:
        cancel = threading.Event()
        cancel.set()
        client = _client(gate.url(), retry=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError) as ei:
            client.probe(cancel_event=cancel)
        assert ei.value.kind == ErrorKind.CANCELLED
        time.sleep(0.05)
        assert gate.accepted == 0
    finally:
        gate.close()


# ---------------------------------------------------------------------------
# Credential store
# ---------------------------------------------------------------------------


def test_v1_migration_is_idempotent_and_lossless(tmp_path: Path) -> None:
    path = profiles.store_path()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": {
                    "fx1": {
                        "key": KEY,
                        "env_var": "MOONSHOT_API_KEY",
                        "base_url": "https://Models.Example/v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    raw = path.read_bytes()
    first = profiles.resolve_endpoint("fx1")
    assert first == (KEY, "https://models.example/v1/chat/completions")
    assert profiles.resolve_endpoint("fx1") == first
    assert path.read_bytes() == raw
    profiles.set_endpoint("fx1-lite", LITE, "https://lite.example/v1", path=path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 2
    assert loaded["profiles"]["fx1"]["key"] == KEY
    assert profiles.resolve_endpoint("fx1") == first
    profiles.set_endpoint("fx1", KEY, "https://Models.Example/v1")
    migrated = profiles.resolve_endpoint("fx1")
    profiles.set_endpoint("fx1", KEY, migrated[1] if migrated else "")
    assert profiles.resolve_endpoint("fx1") == migrated
    assert profiles.resolve_endpoint("fx1-lite") == (
        LITE,
        "https://lite.example/v1/chat/completions",
    )
    assert not (tmp_path / "unused").exists()


def test_four_processes_do_not_lose_a_profile(tmp_path: Path) -> None:
    config = tmp_path / "cfg"
    config.mkdir()
    keys: dict[str, list[str]] = {"fx1": [], "fx1-lite": []}
    procs: list[subprocess.Popen[bytes]] = []
    for index in range(4):
        model = "fx1" if index < 2 else "fx1-lite"
        key = f"sk-proc-{index}-secret-value"
        keys[model].append(key)
        key_file = tmp_path / f"key-{index}"
        key_file.write_text(key, encoding="utf-8")
        key_file.chmod(0o600)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CHILD,
                    str(config),
                    model,
                    str(key_file),
                    f"https://p{index}.example/v1",
                ],
                cwd="/workspace",
                env=_child_env(config),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    for proc in procs:
        _out, err = proc.communicate(timeout=20)
        assert proc.returncode == 0, err.decode("utf-8", errors="replace")
    store = config / "credentials.json"
    loaded = json.loads(store.read_text(encoding="utf-8"))
    assert set(loaded["profiles"]) == {"fx1", "fx1-lite"}
    for model, entry in loaded["profiles"].items():
        assert entry["api_key"] in keys[model]
        assert entry["base_url"].endswith("/v1/chat/completions")
    assert stat.S_IMODE(store.stat().st_mode) == 0o600


def test_symlink_store_is_refused_and_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim_key = f"sk-victim-{TOKEN}"
    original = json.dumps(
        {
            "schema_version": 2,
            "profiles": {
                "fx1": {
                    "api_key": victim_key,
                    "base_url": "https://victim.example/v1/chat/completions",
                }
            },
        }
    ).encode()
    victim.write_bytes(original)
    parent = tmp_path / "parent"
    parent.mkdir()
    link = parent / "credentials.json"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink") as ei:
        profiles.resolve_endpoint("fx1", path=link)
    _clean(str(ei.value), victim_key, KEY, TOKEN)
    with pytest.raises(ValueError, match="symlink"):
        profiles.set_endpoint("fx1", KEY, "https://h.example/v1", path=link)
    assert link.is_symlink()
    assert victim.read_bytes() == original
    warnings = " ".join(profiles.store_health(link))
    assert "symlink" in warnings
    _clean(warnings, victim_key, KEY)

    real = tmp_path / "real"
    real.mkdir()
    target = real / "credentials.json"
    target.write_bytes(original)
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink") as directory:
        profiles.set_endpoint(
            "fx1", KEY, "https://h.example/v1", path=linked_dir / "credentials.json"
        )
    _clean(str(directory.value), KEY, victim_key)
    assert target.read_bytes() == original
    assert KEY.encode() not in target.read_bytes()


def test_world_writable_config_dir_is_tightened_or_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = tmp_path / "open"
    opened.mkdir()
    os.chmod(opened, 0o777)
    path = profiles.set_endpoint(
        "fx1", KEY, "https://h.example/v1", path=opened / "credentials.json"
    )
    assert stat.S_IMODE(opened.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    _clean(" ".join(profiles.store_health(path)), KEY)

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o777)
    real_chmod = os.chmod

    def boom(target: Path, mode: int) -> None:
        if mode == 0o700:
            raise OSError("chmod blocked")
        real_chmod(target, mode)

    monkeypatch.setattr(profiles.os, "chmod", boom)
    with pytest.raises(ValueError, match="writable") as ei:
        profiles.set_endpoint("fx1", KEY, "https://h.example/v1", path=blocked / "credentials.json")
    _clean(str(ei.value), KEY)
    assert not (blocked / "credentials.json").exists()
    assert stat.S_IMODE(blocked.stat().st_mode) == 0o777


def test_windows_acl_hostile_username_is_one_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "credentials.json"
    path.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("USERNAME", 'ada "evil";calc.exe')
    assert profiles.protect_store_windows(path, runner=runner) is True
    argv, kwargs = calls[0]
    assert isinstance(argv, list)
    assert argv[0] == "icacls"
    assert "cmd" not in argv
    assert "shell" not in kwargs
    assert kwargs.get("shell") is not True
    assert argv[-1] == 'ada "evil";calc.exe:F'
    monkeypatch.setenv("USERNAME", "Ada Lovelace")
    assert profiles.protect_store_windows(path, runner=runner) is True
    assert calls[1][0][-1] == "Ada Lovelace:F"
    monkeypatch.setenv("USERNAME", "ada\r\ncalc")
    calls.clear()
    assert profiles.protect_store_windows(path, runner=runner) is False
    assert calls == []


def test_list_and_describe_never_expose_keys() -> None:
    profiles.set_endpoint("fx1", KEY, "https://stored.example/v1")
    rows = profiles.list_models(environ={profiles.ENV_API_KEY: LITE})
    blob = json.dumps(rows)
    _clean(blob, KEY, LITE)
    assert "api_key" not in blob
    described = normalize_endpoint("https://stored.example/v1").describe()
    _clean(described, KEY)
    info = profiles.resolve_endpoint_info("fx1")
    assert info is not None
    _clean(repr(info), KEY)
    _clean(str(info), KEY)


def test_suggest_base_url_never_invents_a_hostname() -> None:
    assert profiles.suggest_base_url({}) == MOONSHOT_API_URL
    assert "artificialhedge" not in profiles.suggest_base_url({}).lower()
    chosen = profiles.suggest_base_url({profiles.ENV_BASE_URL: "https://only.example/v1"})
    assert chosen == "https://only.example/v1"
    assert "artificialhedge" not in chosen.lower()
    untouched = profiles.suggest_base_url(
        {profiles.ENV_FX1_BASE_URL: "https://fx1-only.example/v1"}
    )
    assert untouched == MOONSHOT_API_URL
    assert "fx1-only.example" not in untouched


def test_shared_env_key_precedence_and_child_process(tmp_path: Path) -> None:
    # Stored profile wins. A per-model URL without a per-model key does not
    # override the shared URL. MOONSHOT_API_KEY alone uses FX1_BASE_URL, else
    # the Moonshot constant. Nothing here invents an Artificial Hedge host.
    profiles.set_endpoint("fx1", KEY, "https://stored.example/v1")
    stored = profiles.resolve_endpoint_info(
        "fx1",
        environ={
            profiles.ENV_API_KEY: LITE,
            profiles.ENV_BASE_URL: "https://shared.example/v1",
            profiles.ENV_FX1_API_KEY: "sk-per-model-key-DDDD",
            profiles.ENV_FX1_BASE_URL: "https://model.example/v1",
        },
    )
    assert stored is not None
    assert stored.api_key == KEY
    assert stored.source == "stored"
    assert stored.completion_url == "https://stored.example/v1/chat/completions"
    shared = profiles.resolve_endpoint_info(
        "fx1-lite",
        environ={
            profiles.ENV_API_KEY: LITE,
            profiles.ENV_FX1_LITE_BASE_URL: "https://should-not-win.example/v1",
        },
    )
    assert shared is not None
    assert shared.api_key == LITE
    assert shared.source == "env:shared"
    assert shared.completion_url == MOONSHOT_API_URL
    explicit = profiles.resolve_endpoint_info(
        "fx1",
        path=tmp_path / "env-only.json",
        environ={profiles.ENV_API_KEY: LITE, profiles.ENV_BASE_URL: "https://shared.example/v1"},
    )
    assert explicit is not None
    assert explicit.completion_url == "https://shared.example/v1/chat/completions"
    assert "artificialhedge" not in explicit.completion_url
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('MOONSHOT_API_KEY', ''))"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.stdout.strip() == ""
    _clean(proc.stdout + proc.stderr, KEY, LITE)
    assert not (tmp_path / "leak").exists()
