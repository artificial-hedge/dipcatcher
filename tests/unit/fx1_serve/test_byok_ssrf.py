"""Regression coverage for the BYOK outbound-network boundary."""

from __future__ import annotations

import contextlib
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from fx1.serve import backends


def _request(url: str = "http://provider.example/v1/chat/completions") -> urllib.request.Request:
    request = urllib.request.Request(url, data=b"{}", method="POST")
    backends._apply_byok_destination_policy(request)
    return request


def test_private_and_mixed_dns_answers_refuse_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[Any, ...]] = []

    def no_connect(*args: Any, **kwargs: Any) -> None:
        attempts.append(args)
        raise AssertionError("a prohibited destination must not be contacted")

    monkeypatch.delenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, raising=False)
    monkeypatch.setattr(socket, "create_connection", no_connect)
    for answers in (
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
        [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 80)),
        ],
    ):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, _answers=answers, **k: _answers)
        with (
            pytest.raises(urllib.error.URLError, match="prohibited network address"),
            backends._openai_urlopen(_request(), timeout_s=1),
        ):
            pass
    assert attempts == []


def test_private_opt_in_is_strict_and_never_admits_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, "true")
    assert backends._byok_address_allowed(  # noqa: SLF001 - security policy unit boundary
        backends._normalized_address("10.1.2.3"), allow_private=True
    )
    assert backends._byok_address_allowed(
        backends._normalized_address("127.0.0.1"), allow_private=True
    )
    assert backends._byok_address_allowed(
        backends._normalized_address("fd00::1"), allow_private=True
    )
    for raw in ("169.254.169.254", "100.64.0.1", "192.0.2.1", "::", "ff02::1"):
        assert not backends._byok_address_allowed(
            backends._normalized_address(raw), allow_private=True
        )


def test_pinned_transport_validates_all_answers_then_uses_one_numeric_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions: list[tuple[str, int]] = []
    connections: list[tuple[str, int, str, float]] = []
    sent: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    class Response:
        status = 200
        reason = "OK"
        headers: dict[str, str] = {}

        def close(self) -> None:
            pass

    class Connection:
        def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
            connections.append((host, port, address, timeout))

        def request(
            self, method: str, target: str, body: bytes | None, headers: dict[str, str]
        ) -> None:
            sent.append((method, target, body, headers))

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            pass

    def resolve(host: str, port: int, **kwargs: Any) -> list[tuple[Any, ...]]:
        resolutions.append((host, port))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(backends, "_ByokPinnedHTTPConnection", Connection)
    response, connection = backends._open_byok_pinned(
        urllib.request.Request(
            "http://provider.example:8080/v1/chat/completions?x=1",
            data=b"payload",
            headers={"Authorization": "Bearer secret"},
            method="POST",
        ),
        timeout_s=3,
        allow_private=False,
    )
    assert response.status == 200
    connection.close()
    assert resolutions == [("provider.example", 8080)]
    assert connections == [("provider.example", 8080, "93.184.216.34", 3)]
    assert sent == [
        ("POST", "/v1/chat/completions?x=1", b"payload", {"Authorization": "Bearer secret"})
    ]


def test_byok_policy_reaches_all_five_provider_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    class Response:
        def __init__(self, url: str) -> None:
            self.url = url

        def read(self) -> bytes:
            if self.url.endswith("/tokenize"):
                return b'{"count": 3}'
            if self.url.endswith("/embeddings"):
                return b'{"object":"list","data":[],"model":"embed","usage":{}}'
            return b'{"choices":[{"message":{"content":"ok"}}]}'

        def __iter__(self) -> Iterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            yield b"data: [DONE]\n"

    @contextlib.contextmanager
    def fake_open(request: urllib.request.Request, *, timeout_s: float) -> Iterator[Response]:
        del timeout_s
        seen.append(request)
        yield Response(request.full_url)

    monkeypatch.delenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, raising=False)
    monkeypatch.setattr(backends, "_openai_urlopen", fake_open)
    backend = backends.OpenAICompatBackend(
        base_url="https://provider.example/v1", api_key="secret", model="chat"
    )
    messages = [{"role": "user", "content": "hello"}]
    assert backend.complete(messages) == "ok"
    assert backend.complete_with_tools(messages).content == "ok"
    assert list(backend.stream(messages)) == ["ok"]
    assert backend.embeddings("hello", model="embed").data == ()
    assert backend.count_tokens(messages) == 3
    assert len(seen) == 5
    assert all(getattr(request, "_fx1_byok_public_only", False) for request in seen)
    assert not any(getattr(request, "_fx1_byok_allow_private", False) for request in seen)


def test_non_byok_transport_keeps_existing_urllib_path(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[Any, ...]] = []

    class Response:
        def close(self) -> None:
            pass

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: float) -> Response:
            assert request.full_url == "http://127.0.0.1/private"
            assert timeout == 2
            return Response()

    def build(*handlers: Any) -> Opener:
        built.append(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build)
    with backends._openai_urlopen(
        urllib.request.Request("http://127.0.0.1/private"), timeout_s=2
    ) as response:
        assert isinstance(response, Response)
    assert len(built) == 1
