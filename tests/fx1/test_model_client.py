"""ModelClient retry, cancel, identity, and error taxonomy, plus the stub server."""

from __future__ import annotations

import contextlib
import http.client
import json
import socket
import ssl
import threading
import time
from collections.abc import Mapping
from email.utils import formatdate
from typing import Any

import pytest

from fx1.interactive.model_client import (
    ErrorKind,
    ModelClient,
    ModelError,
    ModelIdentity,
    RetryPolicy,
    choose_proxy,
    parse_retry_after,
    retry_backoff_delay,
)
from fx1.serve.backends import SamplingParams, ToolCompletion
from tests.fx1.stub_model_server import Reply, StubModelServer

KEY_FX1 = "key-fx1-xxxxxxxx"
KEY_LITE = "key-lite-yyyyyyyy"
KEYS = {"fx1": KEY_FX1, "fx1-lite": KEY_LITE}
SHAPE_SECRET = "sk-testsecretkeyvalue"
_NOW = 1_700_000_000.0


def _msg(text: str = "hello") -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _openai_error(
    message: str, code: str, err_type: str = "invalid_request_error"
) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "code": code}}


def _tool_call() -> dict[str, Any]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q": 1}'},
    }


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def srv() -> Any:
    server = StubModelServer(keys=KEYS)
    with server:
        yield server


def _client(srv: StubModelServer, **overrides: Any) -> tuple[ModelClient, list[float]]:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds))

    params: dict[str, Any] = {
        "api_key": KEY_FX1,
        "model": "fx1",
        "api_url": srv.url,
        "timeout_s": 5.0,
        "retry": RetryPolicy(max_attempts=2, base_delay_s=0.05, max_delay_s=1.0, jitter=0.0),
        "sleep": sleep,
        "proxies": lambda: {},
        "bypass": lambda _host: False,
    }
    params.update(overrides)
    return ModelClient(**params), sleeps


def _events() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    found: list[tuple[str, dict[str, Any]]] = []

    def emit(name: str, fields: Mapping[str, Any]) -> None:
        found.append((name, dict(fields)))

    return found, emit


def _assert_clean(text: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in text


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _ConnectProxy:
    """Minimal HTTP CONNECT proxy. Records only the preamble, then pipes bytes."""

    def __init__(self) -> None:
        self.preamble = b""
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind(("127.0.0.1", 0))
        self._listen.listen(1)
        self._listen.settimeout(2.0)
        self.port = int(self._listen.getsockname()[1])
        self._thread = threading.Thread(target=self._run, name="fx1-test-proxy", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._listen.close()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            client, _addr = self._listen.accept()
        except OSError:
            return
        try:
            self._tunnel(client)
        finally:
            client.close()

    def _tunnel(self, client: socket.socket) -> None:
        self.preamble = _read_headers(client)
        target = _connect_target(self.preamble)
        if target is None:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return
        upstream = socket.create_connection(target, timeout=2.0)
        try:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _pipe(client, upstream)
        finally:
            upstream.close()


def _read_headers(sock: socket.socket) -> bytes:
    sock.settimeout(2.0)
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 8192:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def _connect_target(preamble: bytes) -> tuple[str, int] | None:
    line = preamble.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
    parts = line.split(" ")
    if len(parts) < 2 or parts[0].upper() != "CONNECT" or ":" not in parts[1]:
        return None
    host, _, port_s = parts[1].rpartition(":")
    try:
        return host, int(port_s)
    except ValueError:
        return None


def _pipe(left: socket.socket, right: socket.socket) -> None:
    done = threading.Event()

    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while not done.is_set():
                src.settimeout(0.2)
                try:
                    data = src.recv(65536)
                except TimeoutError:
                    continue
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            done.set()

    threads = (
        threading.Thread(target=pump, args=(left, right), daemon=True),
        threading.Thread(target=pump, args=(right, left), daemon=True),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    done.set()


def _forbid_sleep(seconds: float) -> None:
    raise AssertionError(f"unexpected retry sleep {seconds}")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _bad_tls_peer() -> tuple[int, threading.Thread, socket.socket]:
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind(("127.0.0.1", 0))
    listen.listen(1)
    listen.settimeout(2.0)
    port = int(listen.getsockname()[1])

    def run() -> None:
        try:
            conn, _addr = listen.accept()
        except OSError:
            return
        try:
            conn.settimeout(1.0)
            with contextlib.suppress(OSError):
                conn.recv(64)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        except OSError:
            pass
        finally:
            conn.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread, listen


# ---------------------------------------------------------------------------
# Policy, proxy decision, URL
# ---------------------------------------------------------------------------


def test_retry_policy_defaults() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.base_delay_s == 0.5
    assert policy.max_delay_s == 8.0
    assert policy.jitter == 0.25
    assert policy.honor_retry_after_up_to_s == 30.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": True}, "max_attempts"),
        ({"base_delay_s": -1}, "base_delay_s"),
        ({"jitter": -0.1}, "jitter"),
        ({"honor_retry_after_up_to_s": -5}, "honor_retry_after"),
    ],
)
def test_retry_policy_rejects_bad_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize(
    ("attempt", "unit", "expected"),
    [
        (1, 0.0, 0.375),
        (1, 0.5, 0.5),
        (1, 1.0, 0.625),
        (2, 0.5, 1.0),
        (3, 0.5, 2.0),
    ],
)
def test_backoff_delay_fractional_jitter(attempt: int, unit: float, expected: float) -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0, jitter=0.25)
    assert retry_backoff_delay(policy, attempt, unit) == pytest.approx(expected)


def test_backoff_delay_caps_and_zero_jitter() -> None:
    capped = RetryPolicy(base_delay_s=5.0, max_delay_s=8.0, jitter=0.0)
    assert retry_backoff_delay(capped, 1, 0.2) == pytest.approx(5.0)
    assert retry_backoff_delay(capped, 2, 0.2) == pytest.approx(8.0)
    assert retry_backoff_delay(capped, 4, 0.9) == pytest.approx(8.0)
    plain = RetryPolicy(base_delay_s=0.5, max_delay_s=8.0, jitter=0.0)
    assert retry_backoff_delay(plain, 3, 0.4) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("-1", None),
        ("nan", None),
        ("0", 0.0),
        ("1.5", 1.5),
        (" 2 ", 2.0),
    ],
)
def test_parse_retry_after_seconds(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value, now=_NOW) == expected


def test_parse_retry_after_http_date() -> None:
    header = formatdate(timeval=_NOW + 2, usegmt=True)
    assert parse_retry_after(header, now=_NOW) == pytest.approx(2.0)
    past = formatdate(timeval=_NOW - 30, usegmt=True)
    assert parse_retry_after(past, now=_NOW) == 0.0
    assert parse_retry_after("not-a-date", now=_NOW) is None


def _bypass_loopback(host: str) -> bool:
    return host == "127.0.0.1"


def test_choose_proxy_bypass_fallback_and_userinfo() -> None:
    bypass = _bypass_loopback
    proxies = {"https": "http://user:pw@proxy.example:8080", "http": "http://proxy.example:8888"}
    assert choose_proxy("api.example.com", "https", proxies=proxies, bypass=bypass) == (
        "proxy.example",
        8080,
    )
    assert choose_proxy("api.example.com", "http", proxies=proxies, bypass=bypass) == (
        "proxy.example",
        8888,
    )
    assert choose_proxy("127.0.0.1", "http", proxies=proxies, bypass=bypass) is None
    assert choose_proxy(
        "api.example.com",
        "https",
        proxies={"http": "http://proxy.example:9"},
        bypass=lambda _host: False,
    ) == ("proxy.example", 9)
    assert (
        choose_proxy(
            "api.example.com",
            "https",
            proxies={"https": "socks5://proxy.example:1080"},
            bypass=lambda _host: False,
        )
        is None
    )
    assert choose_proxy(
        "api.example.com",
        "http",
        proxies={"http": "http://proxy.example"},
        bypass=lambda _host: False,
    ) == ("proxy.example", 80)
    assert (
        choose_proxy(
            "api.example.com",
            "https",
            proxies={"https": "http://proxy.example:notaport"},
            bypass=lambda _host: False,
        )
        is None
    )


def test_choose_proxy_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("http_proxy", "http://proxy.example:8080")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    assert choose_proxy("127.0.0.1", "http") is None
    assert choose_proxy("api.example.com", "http") == ("proxy.example", 8080)


def test_rejects_unsafe_urls_without_echoing_them() -> None:
    secret = "supersecretvalue"
    cases = (
        "http://example.com/v1/chat/completions",
        f"https://user:{secret}@example.com/v1/chat/completions",
        f"https://example.com/v1/chat/completions?api_key={secret}",
    )
    for url in cases:
        with pytest.raises(RuntimeError) as ei:
            ModelClient(api_key=KEY_FX1, api_url=url, proxies=lambda: {})
        text = f"{ei.value} {ei.value!r}"
        _assert_clean(text, secret, url)


def test_accepts_default_and_ipv6_loopback_without_connecting() -> None:
    hosted = ModelClient(api_key=KEY_FX1, proxies=lambda: {})
    assert hosted.__repr__() == "ModelClient(model='fx1', host='api.moonshot.ai')"
    _assert_clean(repr(hosted), KEY_FX1, "/v1/chat/completions")
    loop = ModelClient(
        api_key=KEY_FX1, api_url="http://[::1]/v1/chat/completions", proxies=lambda: {}
    )
    assert "::1" in repr(loop)
    _assert_clean(repr(loop), KEY_FX1)


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan")])
def test_timeout_must_be_positive(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        ModelClient(api_key=KEY_FX1, api_url="https://example.com/v1", timeout_s=timeout_s)


def test_response_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        ModelClient(
            api_key=KEY_FX1,
            api_url="https://example.com/v1",
            proxies=lambda: {},
            max_response_bytes=0,
        )


def test_missing_and_empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        ModelClient(api_url="https://example.com/v1", proxies=lambda: {})
    text = ""
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY") as ei:
        ModelClient(api_key="   ", api_url="https://example.com/v1", proxies=lambda: {})
    text = str(ei.value)
    assert "https://" not in text


def test_env_key_and_base_url(srv: StubModelServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", KEY_LITE)
    monkeypatch.setenv("FX1_BASE_URL", srv.url)
    client = ModelClient(model="fx1-lite", proxies=lambda: {}, retry=RetryPolicy(max_attempts=1))
    assert client.complete(_msg()) == "stub fx1-lite reply"
    assert srv.requests[0].auth_ok is True
    assert srv.requests[0].auth_fingerprint == "...yyyy"


# ---------------------------------------------------------------------------
# Happy path, identity, usage, drop-in shape
# ---------------------------------------------------------------------------


def test_complete_echo_and_default_temperature(srv: StubModelServer) -> None:
    client, sleeps = _client(srv)
    assert client.complete(_msg("hello")) == "stub fx1 reply"
    body = srv.requests[0].body or {}
    assert body["temperature"] == 0.0
    assert "max_tokens" not in body
    assert body["model"] == "fx1"
    assert sleeps == []
    assert srv.requests[0].method == "POST"
    assert srv.requests[0].path.endswith("/v1/chat/completions")
    assert srv.requests[0].n_messages == 1
    assert srv.requests[0].has_tools is False
    assert KEY_FX1 not in json.dumps(srv.requests[0].body)


def test_sampling_tools_and_stop(srv: StubModelServer) -> None:
    srv.queue(Reply.ok("hello END world", served_model="fx1", usage={"completion_tokens": 3}))
    client, _sleeps = _client(srv)
    text = client.complete(
        _msg(),
        sampling=SamplingParams(temperature=0.2, seed=7, max_tokens=12, stop=("END",)),
    )
    assert text == "hello "
    body = srv.requests[0].body or {}
    assert body["temperature"] == 0.2
    assert body["seed"] == 7
    assert body["max_tokens"] == 12
    assert body["stop"] == ["END"]


def test_drop_in_factory_and_tool_calls(srv: StubModelServer) -> None:
    call = _tool_call()
    srv.queue(
        Reply.ok(
            None,
            tool_calls=[call],
            served_model="fx1",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            system_fingerprint="fp-1",
            headers={"X-Request-Id": "req-9"},
        )
    )
    events, emit = _events()

    def factory(api_key: str, model: str, api_url: str) -> ModelClient:
        built, _sleeps = _client(
            srv,
            api_key=api_key,
            model=model,
            api_url=api_url,
            emit=emit,
            retry=RetryPolicy(max_attempts=1),
        )
        return built

    backend = factory(api_key=KEY_FX1, model="fx1", api_url=srv.url)
    tools = [{"type": "function", "function": {"name": "lookup", "description": "SENTINEL_TOOL"}}]
    completion = backend.complete_with_tools(
        _msg("SENTINEL_PROMPT"),
        sampling=None,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    assert isinstance(completion, ToolCompletion)
    assert completion.content is None
    assert completion.finish_reason == "tool_calls"
    assert completion.tool_calls is not None
    assert completion.tool_calls[0]["id"] == "call_1"
    assert completion.tool_calls[0]["function"]["name"] == "lookup"
    body = srv.requests[0].body or {}
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False
    assert srv.requests[0].has_tools is True
    identity = backend.last_identity
    assert identity is not None
    assert identity.matched is True
    assert identity.request_id == "req-9"
    assert identity.system_fingerprint == "fp-1"
    assert identity.served == "fx1"
    blob = json.dumps(events)
    _assert_clean(blob, "SENTINEL_PROMPT", "SENTINEL_TOOL", KEY_FX1)
    names = [name for name, _fields in events]
    assert names == ["model.request", "model.response"]
    request = events[0][1]
    assert request == {"model": "fx1", "host": identity.host, "n_messages": 1, "n_tools": 1}
    response = events[1][1]
    assert response["status"] == 200
    assert response["served_model"] == "fx1"
    assert response["request_id"] == "req-9"
    assert response["usage"]["total_tokens"] == 6


def test_both_models_use_distinct_keys(srv: StubModelServer) -> None:
    full, _a = _client(srv, model="fx1", api_key=KEY_FX1, retry=RetryPolicy(max_attempts=1))
    lite, _b = _client(srv, model="fx1-lite", api_key=KEY_LITE, retry=RetryPolicy(max_attempts=1))
    wrong, _c = _client(srv, model="fx1-lite", api_key=KEY_FX1, retry=RetryPolicy(max_attempts=1))
    assert full.complete(_msg()) == "stub fx1 reply"
    assert lite.complete(_msg()) == "stub fx1-lite reply"
    with pytest.raises(ModelError) as ei:
        wrong.complete(_msg())
    assert ei.value.kind == ErrorKind.AUTH
    assert srv.requests[0].auth_fingerprint == "...xxxx"
    assert srv.requests[1].auth_fingerprint == "...yyyy"
    assert srv.requests[2].auth_ok is False
    assert srv.requests[2].auth_fingerprint == "...xxxx"


def test_ping_records_one_token_probe(srv: StubModelServer) -> None:
    client, _sleeps = _client(srv)
    identity = client.ping()
    assert isinstance(identity, ModelIdentity)
    assert identity.matched is True
    assert identity.request_id == "stub-1"
    assert identity.system_fingerprint == "stub"
    body = srv.requests[0].body or {}
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0.0
    assert client.last_identity == identity


def test_identity_mismatch_does_not_raise_or_relabel(srv: StubModelServer) -> None:
    srv.queue(Reply.ok("from-other", served_model="fx1-lite", headers={"x-request-id": "rid"}))
    events, emit = _events()
    client, _sleeps = _client(srv, emit=emit)
    assert client.complete(_msg()) == "from-other"
    identity = client.last_identity
    assert identity is not None
    assert identity.requested == "fx1"
    assert identity.served == "fx1-lite"
    assert identity.matched is False
    mismatch = [fields for name, fields in events if name == "model.identity_mismatch"]
    assert mismatch == [
        {
            "requested": "fx1",
            "served": "fx1-lite",
            "host": identity.host,
            "request_id": "rid",
        }
    ]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-request-id": "alpha", "request-id": "beta"}, "alpha"),
        ({"request-id": "beta"}, "beta"),
        ({"openai-request-id": "gamma"}, "gamma"),
        ({"X-Request-Id": "  delta  "}, "delta"),
    ],
)
def test_request_id_header_priority(
    srv: StubModelServer, headers: dict[str, str], expected: str
) -> None:
    srv.queue(Reply.ok("ok", served_model="fx1", headers=headers))
    client, _sleeps = _client(srv)
    client.complete(_msg())
    identity = client.last_identity
    assert identity is not None
    assert identity.request_id == expected


def test_missing_model_field_leaves_matched_unknown(srv: StubModelServer) -> None:
    srv.queue(Reply.ok("plain"))
    client, _sleeps = _client(srv)
    assert client.complete(_msg()) == "plain"
    identity = client.last_identity
    assert identity is not None
    assert identity.served is None
    assert identity.matched is None


def test_usage_filters_non_ints_and_accumulates(srv: StubModelServer) -> None:
    srv.queue(
        Reply(
            status=200,
            body=json.dumps(
                {
                    "model": "fx1",
                    "choices": [{"message": {"content": "one"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": True,
                        "completion_tokens": 2,
                        "total_tokens": 2.5,
                        "note": "x",
                    },
                }
            ).encode(),
        )
    )
    srv.queue(Reply.ok("two", served_model="fx1"))  # no usage block
    client, _sleeps = _client(srv)
    client.complete(_msg())
    assert client.last_usage == {"completion_tokens": 2}
    assert client.total_usage == {"completion_tokens": 2}
    client.complete(_msg())
    assert client.last_usage is None
    assert client.total_usage == {"completion_tokens": 2}
    snapshot = client.total_usage
    snapshot["completion_tokens"] = 0
    assert client.total_usage["completion_tokens"] == 2
    usage_snapshot = client.last_usage
    assert usage_snapshot is None


def test_logprobs_and_non_string_finish_reason(srv: StubModelServer) -> None:
    srv.queue(
        Reply(
            status=200,
            body=json.dumps(
                {
                    "model": "fx1",
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": 1,
                            "logprobs": {"tokens": ["ok"]},
                        }
                    ],
                }
            ).encode(),
        )
    )
    client, _sleeps = _client(srv)
    completion = client.complete_with_tools(_msg())
    assert completion.content == "ok"
    assert completion.finish_reason is None
    assert completion.logprobs == {"tokens": ["ok"]}


def test_complete_rejects_null_content(srv: StubModelServer) -> None:
    srv.queue(Reply.ok(None, tool_calls=[_tool_call()], served_model="fx1"))
    client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=2))
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    assert ei.value.kind == ErrorKind.MALFORMED
    assert ei.value.retryable is False
    assert sleeps == []
    assert len(srv.requests) == 1
    assert client.total_usage == {}


def test_concurrent_calls_share_usage(srv: StubModelServer) -> None:
    client, _sleeps = _client(srv, retry=RetryPolicy(max_attempts=1))
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            barrier.wait(timeout=2.0)
            for _ in range(5):
                assert client.complete(_msg()) == "stub fx1 reply"
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert client.total_usage["prompt_tokens"] == 10
    assert client.total_usage["completion_tokens"] == 10
    assert client.total_usage["total_tokens"] == 20
    assert client.last_identity is not None
    assert client.last_identity.matched is True


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


def _require(
    srv: StubModelServer, reply: Reply, **overrides: Any
) -> tuple[ModelError, list[float]]:
    if "retry" not in overrides:
        overrides["retry"] = RetryPolicy(max_attempts=1)
    srv.queue(reply)
    client, sleeps = _client(srv, **overrides)
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    return ei.value, sleeps


def test_auth_permission_and_quota_are_not_retried(srv: StubModelServer) -> None:
    wrong, sleeps = _client(srv, api_key="key-fx1-wrongkey", retry=RetryPolicy(max_attempts=3))
    with pytest.raises(ModelError) as auth:
        wrong.complete(_msg())
    assert auth.value.kind == ErrorKind.AUTH
    assert auth.value.status == 401
    assert auth.value.retryable is False
    assert "`/keys set fx1`" in auth.value.hint
    assert sleeps == []
    assert len(srv.requests) == 1
    _assert_clean(str(auth.value), "key-fx1-wrongkey", "/v1/chat/completions")
    _assert_clean(repr(auth.value), "key-fx1-wrongkey")

    denied, denied_sleeps = _require(
        srv,
        Reply.error(403, body=_openai_error("forbidden", "permission_denied")),
        retry=RetryPolicy(max_attempts=3, jitter=0),
    )
    assert denied.kind == ErrorKind.PERMISSION
    assert denied.retryable is False
    assert denied_sleeps == []

    quota, quota_sleeps = _require(
        srv,
        Reply.error(
            429,
            body=_openai_error("billing hard limit", "insufficient_quota"),
        ),
        retry=RetryPolicy(max_attempts=3, jitter=0),
    )
    assert quota.kind == ErrorKind.QUOTA
    assert quota.retryable is False
    assert quota_sleeps == []


def test_unknown_model_versus_route(srv: StubModelServer) -> None:
    missing, _sleeps = _client(srv, model="nope", retry=RetryPolicy(max_attempts=1))
    with pytest.raises(ModelError) as unknown:
        missing.complete(_msg())
    assert unknown.value.kind == ErrorKind.UNKNOWN_MODEL
    assert unknown.value.status == 404
    assert "nope" in unknown.value.hint
    route, _route_sleeps = _require(
        srv, Reply.error(404, body=_openai_error("Not found", "not_found"))
    )
    assert route.kind == ErrorKind.ROUTE_NOT_FOUND
    assert "/v1/chat/completions" in route.hint
    bad_model, _model_sleeps = _require(
        srv,
        Reply.error(400, body=_openai_error("The model 'fx1' does not exist", "model_not_found")),
    )
    assert bad_model.kind == ErrorKind.UNKNOWN_MODEL


def test_context_bad_request_and_payment(srv: StubModelServer) -> None:
    context, _a = _require(
        srv,
        Reply.error(
            400,
            body=_openai_error("maximum context length exceeded", "context_length_exceeded"),
        ),
    )
    assert context.kind == ErrorKind.CONTEXT_TOO_LONG
    huge, _b = _require(srv, Reply.error(413, body=_openai_error("payload too large", "too_large")))
    assert huge.kind == ErrorKind.CONTEXT_TOO_LONG
    bad, _c = _require(srv, Reply.error(400, body=_openai_error("bad schema", "invalid_request")))
    assert bad.kind == ErrorKind.BAD_REQUEST
    other, _d = _require(srv, Reply.error(300, body=_openai_error("multiple choices", "ambiguous")))
    assert other.kind == ErrorKind.BAD_REQUEST
    billing, _e = _require(srv, Reply.error(402, body=_openai_error("payment required", "billing")))
    assert billing.kind == ErrorKind.QUOTA
    assert billing.retryable is False


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_not_followed(srv: StubModelServer, status: int) -> None:
    with StubModelServer(keys=KEYS) as target:
        token = "leaked-route-token"
        srv.queue(Reply.redirect(f"{target.base_url}/{token}", status=status))
        client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=2, jitter=0))
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == ErrorKind.ROUTE_NOT_FOUND
        assert ei.value.retryable is False
        assert "redirect" in ei.value.hint
        _assert_clean(str(ei.value), token, KEY_FX1)
        assert target.requests == []
        assert sleeps == []
        assert len(srv.requests) == 1


def test_server_retryable_statuses(srv: StubModelServer) -> None:
    for status in (500, 502, 503, 504):
        srv.queue(Reply.error(status, body=_openai_error("down", "server_error", "server_error")))
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=1, jitter=0),
    )
    seen: list[ErrorKind] = []
    for _status in (500, 502, 503, 504):
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        seen.append(ei.value.kind)
        assert ei.value.retryable is True
        assert ei.value.kind == ErrorKind.SERVER
    assert sleeps == []
    fatal, fatal_sleeps = _require(
        srv,
        Reply.error(501, body=_openai_error("nope", "not_implemented", "server_error")),
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.05, jitter=0),
    )
    assert fatal.kind == ErrorKind.SERVER
    assert fatal.retryable is False
    assert fatal_sleeps == []


def test_retryable_server_then_success_counts_only_success(srv: StubModelServer) -> None:
    srv.queue(Reply.error(503, body=_openai_error("busy", "unavailable", "server_error")))
    srv.queue(
        Reply.ok(
            "later",
            served_model="fx1",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
    )
    events, emit = _events()
    client, sleeps = _client(
        srv,
        emit=emit,
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.25, max_delay_s=8, jitter=0),
        rand=lambda: 0.5,
    )
    assert client.complete(_msg("SENTINEL_PROMPT")) == "later"
    assert sleeps == [pytest.approx(0.25)]
    assert client.total_usage == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    names = [name for name, _fields in events]
    assert names == [
        "model.request",
        "model.error",
        "model.retry",
        "model.request",
        "model.response",
    ]
    assert events[1][1] == {"kind": "server", "status": 503}
    assert events[2][1]["delay_s"] == pytest.approx(0.25)
    _assert_clean(json.dumps(events), "SENTINEL_PROMPT", KEY_FX1)


def test_exact_backoff_schedule(srv: StubModelServer) -> None:
    for _ in range(3):
        srv.queue(Reply.error(503, body=_openai_error("busy", "unavailable", "server_error")))
    srv.queue(Reply.ok("done", served_model="fx1"))
    units = iter((0.0, 0.5, 1.0))
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=4, base_delay_s=0.5, max_delay_s=8.0, jitter=0.25),
        rand=lambda: next(units),
    )
    assert client.complete(_msg()) == "done"
    assert sleeps == [pytest.approx(0.375), pytest.approx(1.0), pytest.approx(2.5)]
    assert len(srv.requests) == 4


def test_capped_backoff_schedule(srv: StubModelServer) -> None:
    for _ in range(3):
        srv.queue(Reply.error(500, body=_openai_error("down", "server_error", "server_error")))
    srv.queue(Reply.ok("done", served_model="fx1"))
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=4, base_delay_s=5.0, max_delay_s=8.0, jitter=0.0),
        rand=lambda: 0.2,
    )
    assert client.complete(_msg()) == "done"
    assert sleeps == [pytest.approx(5.0), pytest.approx(8.0), pytest.approx(8.0)]


def test_max_attempts_bounds_retries(srv: StubModelServer) -> None:
    for _ in range(3):
        srv.queue(Reply.error(503, body=_openai_error("busy", "unavailable", "server_error")))
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.2, max_delay_s=1.0, jitter=0),
    )
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    assert ei.value.kind == ErrorKind.SERVER
    assert ei.value.retryable is True
    assert len(srv.requests) == 3
    assert sleeps == [pytest.approx(0.2), pytest.approx(0.4)]


def test_retry_after_seconds_and_http_date(srv: StubModelServer) -> None:
    srv.queue(
        Reply.error(
            429,
            headers={"Retry-After": "1.5"},
            body=_openai_error("slow down", "rate_limit_exceeded"),
        )
    )
    srv.queue(Reply.ok("ok", served_model="fx1"))
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.05, jitter=0.25),
        rand=lambda: 0.0,
    )
    assert client.complete(_msg()) == "ok"
    assert sleeps == [pytest.approx(1.5)]
    assert client.last_identity is not None

    header = formatdate(timeval=_NOW + 2, usegmt=True)
    srv.queue(
        Reply.error(
            429,
            headers={"Retry-After": header},
            body=_openai_error("slow down", "rate_limit_exceeded"),
        )
    )
    srv.queue(Reply.ok("dated", served_model="fx1"))
    dated, dated_sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=2, base_delay_s=9, jitter=0),
        now=lambda: _NOW,
        rand=lambda: 0.0,
    )
    assert dated.complete(_msg()) == "dated"
    assert dated_sleeps == [pytest.approx(2.0)]


def test_retry_after_past_and_invalid_fall_back(srv: StubModelServer) -> None:
    past = formatdate(timeval=time.time() - 3600, usegmt=True)
    srv.queue(
        Reply.error(
            429,
            headers={"Retry-After": past},
            body=_openai_error("slow down", "rate_limit_exceeded"),
        )
    )
    srv.queue(Reply.ok("now", served_model="fx1"))
    client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=2, base_delay_s=0.4, jitter=0))
    assert client.complete(_msg()) == "now"
    assert sleeps == []

    srv.queue(
        Reply.error(
            429,
            headers={"Retry-After": "-5"},
            body=_openai_error("slow down", "rate_limit_exceeded"),
        )
    )
    srv.queue(Reply.ok("backoff", served_model="fx1"))
    backed, backed_sleeps = _client(
        srv, retry=RetryPolicy(max_attempts=2, base_delay_s=0.3, jitter=0)
    )
    assert backed.complete(_msg()) == "backoff"
    assert backed_sleeps == [pytest.approx(0.3)]


def test_retry_after_above_cap_is_not_auto_retried(srv: StubModelServer) -> None:
    srv.queue(
        Reply.error(
            429,
            headers={"Retry-After": "45"},
            body=_openai_error("slow down", "rate_limit_exceeded"),
        )
    )
    client, sleeps = _client(
        srv,
        retry=RetryPolicy(max_attempts=3, honor_retry_after_up_to_s=30, jitter=0),
    )
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    assert ei.value.kind == ErrorKind.RATE_LIMITED
    assert ei.value.retryable is True
    assert ei.value.retry_after_s == pytest.approx(45)
    assert "45" in ei.value.hint
    assert "30" in ei.value.hint
    assert sleeps == []
    assert len(srv.requests) == 1


def test_408_is_retryable_timeout(srv: StubModelServer) -> None:
    srv.queue(Reply.error(408, body=_openai_error("timeout", "timeout")))
    srv.queue(Reply.ok("recovered", served_model="fx1"))
    client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=2, base_delay_s=0.15, jitter=0))
    assert client.complete(_msg()) == "recovered"
    assert sleeps == [pytest.approx(0.15)]
    err_kind = ErrorKind.TIMEOUT
    assert err_kind.value == "timeout"


@pytest.mark.parametrize(
    "body",
    [
        b"<<<not-json>>>",
        b"{}",
        b'{"choices": []}',
        b'{"choices": [null]}',
        b'{"choices": [{"message": {"content": 1}}]}',
        b'{"choices": [{"message": {"content": null, "tool_calls": "nope"}}]}',
        b'{"choices": [{"message": {"content": null, "tool_calls": [{}]}}]}',
        b'{"choices": [{"message": {"content": "ok"}, "logprobs": []}]}',
        b'{"choices": [{"message": {"content": "ok", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "n", "arguments": {}}}]}}]}',
    ],
)
def test_malformed_payloads_are_not_retried(srv: StubModelServer, body: bytes) -> None:
    srv.queue(Reply.malformed(body))
    client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=3, base_delay_s=0.2, jitter=0))
    with pytest.raises(ModelError) as ei:
        client.complete_with_tools(_msg())
    assert ei.value.kind == ErrorKind.MALFORMED
    assert ei.value.retryable is False
    assert ei.value.status == 200
    assert sleeps == []
    assert "malformed completion" in ei.value.summary


def test_response_body_cap(srv: StubModelServer) -> None:
    marker = "UNIQUE_BODY_MARKER_ZZZ"
    srv.queue(Reply.ok(marker + ("x" * 4000), served_model="fx1"))
    client, sleeps = _client(
        srv,
        max_response_bytes=128,
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.2, jitter=0),
    )
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    assert ei.value.kind == ErrorKind.MALFORMED
    assert "exceeded 128 bytes" in ei.value.summary
    assert ei.value.retryable is False
    _assert_clean(str(ei.value), marker, KEY_FX1)
    assert sleeps == []
    assert len(srv.requests) == 1


def test_error_body_is_capped_and_redacted(srv: StubModelServer) -> None:
    tail = "TAILMARKER9"
    blob = b'{"error":{"message":"' + (b"A" * 70_000) + tail.encode() + b'"}}'
    srv.queue(Reply(status=400, body=blob))
    client, _sleeps = _client(srv, retry=RetryPolicy(max_attempts=1))
    with pytest.raises(ModelError) as capped:
        client.complete(_msg())
    assert capped.value.kind == ErrorKind.BAD_REQUEST
    _assert_clean(str(capped.value), tail)

    echoed = f"rejected {KEY_FX1} and {SHAPE_SECRET}"
    srv.queue(
        Reply.error(400, body={"error": {"message": echoed, "type": "invalid_request_error"}})
    )
    events, emit = _events()
    secret_client, _secret_sleeps = _client(srv, emit=emit, retry=RetryPolicy(max_attempts=1))
    with pytest.raises(ModelError) as leaked:
        secret_client.complete(_msg())
    rendered = " ".join(
        (
            str(leaked.value),
            repr(leaked.value),
            leaked.value.hint,
            leaked.value.summary,
            json.dumps(events),
            repr(secret_client),
        )
    )
    _assert_clean(rendered, KEY_FX1, SHAPE_SECRET)
    assert "[REDACTED]" in leaked.value.summary
    assert leaked.value.kind == ErrorKind.BAD_REQUEST
    assert events[-1][0] == "model.error"
    assert set(events[-1][1]) == {"kind", "status"}


def test_network_refused_and_dns(srv: StubModelServer, monkeypatch: pytest.MonkeyPatch) -> None:
    port = _free_port()
    client, sleeps = _client(
        srv,
        api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
        retry=RetryPolicy(max_attempts=2, base_delay_s=0.05, jitter=0),
    )
    with pytest.raises(ModelError) as refused:
        client.complete(_msg())
    assert refused.value.kind == ErrorKind.NETWORK
    assert refused.value.retryable is True
    assert f"127.0.0.1:{port}" in refused.value.hint
    _assert_clean(str(refused.value), "/v1/chat/completions", KEY_FX1)
    assert sleeps == [pytest.approx(0.05)]

    def fail_dns(*_args: Any, **_kwargs: Any) -> Any:
        raise socket.gaierror(8, "name or service not known")

    monkeypatch.setattr(socket, "create_connection", fail_dns)
    dns, dns_sleeps = _client(
        srv,
        api_url="http://127.0.0.1:9/v1/chat/completions",
        retry=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(ModelError) as lookup:
        dns.complete(_msg())
    assert lookup.value.kind == ErrorKind.NETWORK
    assert "gaierror" in lookup.value.summary
    assert dns_sleeps == []
    _assert_clean(str(lookup.value), KEY_FX1, "name or service not known")


def test_dropped_connection_is_network(srv: StubModelServer) -> None:
    srv.queue(Reply.drop())
    srv.queue(Reply.ok("recovered", served_model="fx1"))
    client, sleeps = _client(srv, retry=RetryPolicy(max_attempts=2, base_delay_s=0.05, jitter=0))
    assert client.complete(_msg()) == "recovered"
    assert sleeps == [pytest.approx(0.05)]
    assert len(srv.requests) == 2


def test_tls_wrong_version_and_cert_error(monkeypatch: pytest.MonkeyPatch) -> None:
    port, thread, listen = _bad_tls_peer()
    try:
        client = ModelClient(
            api_key=KEY_FX1,
            api_url=f"https://127.0.0.1:{port}/v1/chat/completions",
            timeout_s=2.0,
            retry=RetryPolicy(max_attempts=2, base_delay_s=0.2, jitter=0),
            proxies=lambda: {},
            sleep=_forbid_sleep,
        )
        with pytest.raises(ModelError) as ei:
            client.complete(_msg())
        assert ei.value.kind == ErrorKind.TLS
        assert ei.value.retryable is False
        assert "certificate" in ei.value.hint
        _assert_clean(str(ei.value), KEY_FX1, "/v1/chat/completions")
    finally:
        listen.close()
        thread.join(timeout=1.0)

    def boom(self: http.client.HTTPSConnection) -> None:
        del self
        raise ssl.SSLCertVerificationError("certificate verify failed")

    monkeypatch.setattr(http.client.HTTPSConnection, "connect", boom)
    cert = ModelClient(
        api_key=KEY_FX1,
        api_url="https://example.com/v1/chat/completions",
        retry=RetryPolicy(max_attempts=2),
        proxies=lambda: {},
        sleep=_forbid_sleep,
    )
    with pytest.raises(ModelError) as cert_err:
        cert.complete(_msg())
    assert cert_err.value.kind == ErrorKind.TLS
    assert cert_err.value.retryable is False
    assert "example.com" in cert_err.value.hint
    _assert_clean(str(cert_err.value), KEY_FX1)


def test_socket_timeout_is_retryable_and_bounded(srv: StubModelServer) -> None:
    srv.queue(Reply.delay(2.0, Reply.ok("slow", served_model="fx1")))
    client, _sleeps = _client(srv, timeout_s=0.25, retry=RetryPolicy(max_attempts=1))
    started = time.monotonic()
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    elapsed = time.monotonic() - started
    assert ei.value.kind == ErrorKind.TIMEOUT
    assert ei.value.retryable is True
    assert elapsed < 1.0
    assert client.total_usage == {}


# ---------------------------------------------------------------------------
# Cancellation and deadlines
# ---------------------------------------------------------------------------


def test_cancel_aborts_before_dispatch(srv: StubModelServer) -> None:
    cancel = threading.Event()
    cancel.set()
    client, sleeps = _client(srv)
    with pytest.raises(ModelError) as ei:
        client.complete(_msg(), cancel_event=cancel)
    assert ei.value.kind == ErrorKind.CANCELLED
    assert ei.value.retryable is False
    assert srv.requests == []
    assert sleeps == []
    assert client.last_identity is None


def test_deadline_already_passed(srv: StubModelServer) -> None:
    client, sleeps = _client(srv)
    with pytest.raises(ModelError) as ei:
        client.complete(_msg(), deadline=time.monotonic() - 1.0)
    assert ei.value.kind == ErrorKind.DEADLINE
    assert srv.requests == []
    assert sleeps == []


def test_cancel_latency_and_no_usage(srv: StubModelServer) -> None:
    srv.queue(Reply.delay(3.0, Reply.ok("slow", served_model="fx1", usage={"total_tokens": 9})))
    client, _sleeps = _client(srv, timeout_s=2.0, retry=RetryPolicy(max_attempts=1))
    cancel = threading.Event()
    caught: list[BaseException] = []

    def run() -> None:
        try:
            client.complete(_msg(), cancel_event=cancel)
        except BaseException as exc:
            caught.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert srv.wait_for_requests(1, timeout=2.0)
    started = time.monotonic()
    cancel.set()
    worker.join(0.5)
    elapsed = time.monotonic() - started
    assert not worker.is_alive()
    assert elapsed <= 0.5
    assert len(caught) == 1
    assert isinstance(caught[0], ModelError)
    assert caught[0].kind == ErrorKind.CANCELLED
    assert client.total_usage == {}
    assert client.last_identity is None
    assert client.complete(_msg()) == "stub fx1 reply"
    assert client.total_usage["completion_tokens"] == 1
    assert client.last_identity is not None
    assert client.last_identity.served == "fx1"


def test_deadline_latency(srv: StubModelServer) -> None:
    srv.queue(Reply.delay(3.0, Reply.ok("slow", served_model="fx1")))
    client, _sleeps = _client(srv, timeout_s=2.0, retry=RetryPolicy(max_attempts=2, jitter=0))
    started = time.monotonic()
    with pytest.raises(ModelError) as ei:
        client.complete(_msg(), deadline=started + 0.15)
    assert ei.value.kind == ErrorKind.DEADLINE
    assert time.monotonic() - started <= 0.5
    assert client.total_usage == {}
    assert len(srv.requests) == 1


def test_cancel_during_retry_sleep(srv: StubModelServer) -> None:
    srv.queue(Reply.error(503, body=_openai_error("busy", "unavailable", "server_error")))
    srv.queue(Reply.ok("should-not-run", served_model="fx1"))
    cancel = threading.Event()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        cancel.set()

    client, _default = _client(
        srv,
        sleep=sleep,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.5, jitter=0),
    )
    with pytest.raises(ModelError) as ei:
        client.complete(_msg(), cancel_event=cancel)
    assert ei.value.kind == ErrorKind.CANCELLED
    assert len(srv.requests) == 1
    assert sleeps == [pytest.approx(0.05)]
    assert client.total_usage == {}


def test_deadline_blocks_backoff(srv: StubModelServer) -> None:
    srv.queue(Reply.error(429, body=_openai_error("slow down", "rate_limit_exceeded")))
    clock = _Clock()
    client, _sleeps = _client(
        srv,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0.5, jitter=0),
    )
    with pytest.raises(ModelError) as ei:
        client.complete(_msg(), deadline=0.1)
    assert ei.value.kind == ErrorKind.DEADLINE
    assert clock.sleeps == []
    assert len(srv.requests) == 1
    assert client.last_usage is None


def test_ping_honors_cancel(srv: StubModelServer) -> None:
    cancel = threading.Event()
    cancel.set()
    client, _sleeps = _client(srv)
    with pytest.raises(ModelError) as ei:
        client.ping(cancel_event=cancel)
    assert ei.value.kind == ErrorKind.CANCELLED
    assert srv.requests == []


# ---------------------------------------------------------------------------
# Stub server contract (reused by other suites)
# ---------------------------------------------------------------------------


def test_stub_queue_then_builtin_default(srv: StubModelServer) -> None:
    srv.queue(Reply.ok("one", served_model="fx1"))
    srv.queue(Reply.ok("two", served_model="fx1"))
    client, _sleeps = _client(srv, retry=RetryPolicy(max_attempts=1))
    assert client.complete(_msg()) == "one"
    assert client.complete(_msg()) == "two"
    assert client.complete(_msg()) == "stub fx1 reply"
    assert [record.model for record in srv.requests] == ["fx1", "fx1", "fx1"]


def test_stub_default_handler_and_recovery(srv: StubModelServer) -> None:
    def boom(_request: Any) -> Reply:
        raise RuntimeError("handler blew up")

    srv.set_default(boom)
    client, _sleeps = _client(srv, retry=RetryPolicy(max_attempts=1))
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    assert ei.value.kind == ErrorKind.SERVER
    assert srv.last_fault == "RuntimeError"
    srv.set_default(None)
    assert client.complete(_msg()) == "stub fx1 reply"


def test_stub_close_interrupts_delay() -> None:
    server = StubModelServer(keys=KEYS)
    server.start()
    try:
        server.queue(Reply.delay(5.0, Reply.ok("slow", served_model="fx1")))
        holder: list[BaseException] = []

        def hit() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=2.0)
            try:
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                conn.getresponse()
            except Exception as exc:
                holder.append(exc)
            finally:
                conn.close()

        worker = threading.Thread(target=hit, daemon=True)
        worker.start()
        assert server.wait_for_requests(1, timeout=2.0)
        started = time.monotonic()
        server.close()
        assert time.monotonic() - started < 0.75
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    finally:
        server.close()


def test_stub_url_before_start_and_relative_import() -> None:
    server = StubModelServer(keys=KEYS)
    with pytest.raises(RuntimeError, match="not running"):
        _ = server.url
    import importlib

    imported = importlib.import_module(".stub_model_server", package="tests.fx1")
    assert imported.StubModelServer is StubModelServer
    assert imported.Reply is Reply


def test_stub_parallel_requests() -> None:
    with StubModelServer(keys=KEYS) as server:
        barrier = threading.Barrier(2)
        found: list[str] = []

        def hit(model: str, key: str) -> None:
            barrier.wait(timeout=2.0)
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=2.0)
            body = json.dumps(
                {"model": model, "messages": [{"role": "user", "content": "x"}]}
            ).encode()
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode())
            found.append(payload["choices"][0]["message"]["content"])
            conn.close()

        threads = [
            threading.Thread(target=hit, args=("fx1", KEY_FX1)),
            threading.Thread(target=hit, args=("fx1-lite", KEY_LITE)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)
        assert sorted(found) == ["stub fx1 reply", "stub fx1-lite reply"]
        assert len(server.requests) == 2
        assert all(record.auth_ok for record in server.requests)


def test_connect_proxy_does_not_see_authorization(srv: StubModelServer) -> None:
    proxy = _ConnectProxy()
    try:
        client, _sleeps = _client(
            srv,
            proxies=lambda: {"http": f"http://127.0.0.1:{proxy.port}"},
            bypass=lambda _host: False,
            retry=RetryPolicy(max_attempts=1),
            timeout_s=3.0,
        )
        assert client.complete(_msg()) == "stub fx1 reply"
        assert b"CONNECT " in proxy.preamble
        assert b"Authorization" not in proxy.preamble
        assert KEY_FX1.encode() not in proxy.preamble
        assert srv.requests[0].auth_ok is True
        assert srv.requests[0].path.endswith("/v1/chat/completions")
    finally:
        proxy.close()


def test_repr_and_error_share_no_key(srv: StubModelServer) -> None:
    client, _sleeps = _client(srv, api_key=KEY_FX1)
    _assert_clean(repr(client), KEY_FX1, "/v1/")
    srv.queue(Reply.error(401, body=_openai_error(f"bad {KEY_FX1}", "invalid_api_key")))
    with pytest.raises(ModelError) as ei:
        client.complete(_msg())
    _assert_clean(str(ei.value), KEY_FX1)
    _assert_clean(repr(ei.value), KEY_FX1)
    _assert_clean(ei.value.hint, KEY_FX1)
    assert ei.value.model == "fx1"
    assert ei.value.host.startswith("127.0.0.1:")
    assert str(ei.value).startswith(f"fx1 @ {ei.value.host}: ")
    assert " — " in str(ei.value)
