"""Fuzz the shared body cap and header checks on both HTTP apps."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.api.app import app as lab_app
from quant_fund.api.http_controls import MAX_BODY_BYTES, MAX_HEADER_COUNT
from quant_fund.api.research_api import ResearchApiSettings, create_app

_AsgiApp = Callable[..., Any]


def _research_app(tmp_path: Any) -> _AsgiApp:
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
    )
    return create_app(settings)


def _invoke(
    app: _AsgiApp,
    *,
    method: str = "POST",
    path: str = "/health",
    headers: list[tuple[bytes, bytes]] | None = None,
    chunks: list[bytes] | None = None,
    client: tuple[str, int] = ("198.51.100.60", 9),
) -> tuple[int, bytes, list[tuple[bytes, bytes]]]:
    raw_headers = headers if headers is not None else [(b"host", b"example.com")]
    parts = chunks if chunks is not None else [b""]
    messages = [
        {"type": "http.request", "body": part, "more_body": index < len(parts) - 1}
        for index, part in enumerate(parts)
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": client,
        "server": ("127.0.0.1", 80),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if messages:
            message: dict[str, Any] = messages.pop(0)
            return message
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response_headers = next(
        message["headers"] for message in sent if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(status), body, list(response_headers)


def _assert_clean(status: int, body: bytes, response_headers: list[tuple[bytes, bytes]]) -> None:
    assert status < 500 or json.loads(body) == {"detail": "internal error"}
    assert b"Traceback" not in body
    assert b"/workspace" not in body
    for name, value in response_headers:
        assert b"\r" not in name and b"\n" not in name
        assert b"\r" not in value and b"\n" not in value
        assert b"x-injected" not in name.lower()


def test_chunked_body_over_the_cap_is_413() -> None:
    chunks = [b"y" * 1024 for _ in range(64)] + [b"y" * 1024]
    status, body, headers = _invoke(lab_app, chunks=chunks)
    assert status == 413
    assert json.loads(body) == {"detail": "request body exceeds the 64 KiB limit"}
    _assert_clean(status, body, headers)


def test_slowloris_style_chunks_stop_at_the_cap() -> None:
    # Empty reads must not reset the counter. The last byte crosses the cap.
    chunks = [b"z" * 1024] * 64 + [b""] * 40 + [b"z"]
    status, body, headers = _invoke(lab_app, chunks=chunks)
    assert status == 413
    assert b"Traceback" not in body
    _assert_clean(status, body, headers)


def test_lying_content_length_cannot_hide_the_body(tmp_path: Any) -> None:
    small = [(b"host", b"example.com"), (b"content-length", b"4")]
    status, _body, _headers = _invoke(lab_app, headers=small, chunks=[b"x" * (MAX_BODY_BYTES + 5)])
    assert status == 413
    huge = [(b"host", b"example.com"), (b"content-length", b"999999")]
    status, body, _headers = _invoke(lab_app, headers=huge, chunks=[b"{}"])
    assert status == 413
    assert b"Traceback" not in body
    research = _research_app(tmp_path)
    status, _body, _headers = _invoke(
        research, headers=small, chunks=[b"q" * 100, b"q" * MAX_BODY_BYTES]
    )
    assert status == 413


def test_header_injection_is_rejected(tmp_path: Any) -> None:
    headers = [
        (b"host", b"example.com"),
        (b"x-api-key", b"abc\r\nX-Injected: pwned"),
    ]
    for app in (lab_app, _research_app(tmp_path)):
        status, body, response_headers = _invoke(app, method="GET", headers=headers, chunks=[b""])
        assert status == 400
        assert json.loads(body) == {"detail": "invalid header"}
        assert b"pwned" not in body
        _assert_clean(status, body, response_headers)


def test_huge_header_count_is_rejected(tmp_path: Any) -> None:
    extra = [(f"x-h-{index}".encode(), b"v") for index in range(MAX_HEADER_COUNT + 10)]
    headers = [(b"host", b"example.com"), *extra]
    for app in (lab_app, _research_app(tmp_path)):
        status, body, response_headers = _invoke(app, method="GET", headers=headers, chunks=[b""])
        assert status == 431
        assert json.loads(body) == {"detail": "too many headers"}
        _assert_clean(status, body, response_headers)


def test_body_under_the_cap_still_reaches_the_route() -> None:
    status, body, headers = _invoke(lab_app, method="GET", chunks=[b""])
    assert status == 200
    assert b"Traceback" not in body
    _assert_clean(status, body, headers)


@settings(max_examples=20, deadline=None)
@given(sizes=st.lists(st.integers(min_value=0, max_value=8000), min_size=1, max_size=20))
def test_random_chunks_follow_the_cap(sizes: list[int]) -> None:
    chunks = [b"a" * size for size in sizes]
    status, body, headers = _invoke(lab_app, chunks=chunks)
    if sum(sizes) > MAX_BODY_BYTES:
        assert status == 413
    else:
        assert status != 413
        assert status < 500
    _assert_clean(status, body, headers)


@settings(max_examples=15, deadline=None)
@given(
    declared=st.integers(min_value=0, max_value=200_000),
    size=st.integers(min_value=0, max_value=90_000),
)
def test_random_content_length_cannot_outrank_the_bytes(declared: int, size: int) -> None:
    headers = [(b"host", b"example.com"), (b"content-length", str(declared).encode())]
    status, body, response_headers = _invoke(lab_app, headers=headers, chunks=[b"b" * size])
    if declared > MAX_BODY_BYTES or size > MAX_BODY_BYTES:
        assert status == 413
    else:
        assert status != 413
    _assert_clean(status, body, response_headers)


@settings(max_examples=10, deadline=None)
@given(count=st.integers(min_value=0, max_value=200))
def test_random_header_counts(count: int) -> None:
    headers = [(b"host", b"example.com")]
    headers.extend((f"x-{index}".encode(), b"v") for index in range(count))
    status, body, response_headers = _invoke(lab_app, method="GET", headers=headers, chunks=[b""])
    if len(headers) > MAX_HEADER_COUNT:
        assert status == 431
    else:
        assert status == 200
    assert b"Traceback" not in body
    _assert_clean(status, body, response_headers)
