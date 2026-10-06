"""Focused request-ingress buffering regressions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.requests import Request

from fx1.serve.api import _MAX_BODY_BYTES, _buffer_request_body


def _chunked_request(chunks: list[bytes]) -> tuple[Request, list[bytes]]:
    pending: Iterator[bytes] = iter(chunks)
    consumed: list[bytes] = []

    async def receive() -> dict[str, Any]:
        chunk = next(pending)
        consumed.append(chunk)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": len(consumed) < len(chunks),
        }

    return (
        Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
            },
            receive,
        ),
        consumed,
    )


@pytest.mark.anyio
async def test_body_buffer_is_reusable_for_downstream_parsing() -> None:
    request, consumed = _chunked_request([b'{"model":', b'"fx-1"}'])

    size, body = await _buffer_request_body(request)

    assert size == len(body)
    assert body == b'{"model":"fx-1"}'
    assert consumed == [b'{"model":', b'"fx-1"}']
    assert await request.body() == body


@pytest.mark.anyio
async def test_oversize_stream_refuses_at_first_excess_chunk() -> None:
    at_cap = b"x" * _MAX_BODY_BYTES
    request, consumed = _chunked_request([at_cap, b"!", b"unread attacker tail"])

    size, body = await _buffer_request_body(request)

    assert size == _MAX_BODY_BYTES + 1
    assert body == b""
    assert consumed == [at_cap, b"!"]
