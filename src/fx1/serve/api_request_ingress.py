"""Request ingress guards for the fx-1 HTTP API (body cap, singleton headers)."""

from __future__ import annotations

import io
import re
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from fx1.serve.anthropic_compat import anthropic_error_body
from fx1.serve.openai_compat import is_openai_path, openai_error_body

MAX_BODY_BYTES = 1 << 20

_SINGLETON_REQUEST_HEADERS = frozenset(
    {
        b"authorization",
        b"content-length",
        b"content-type",
        b"host",
        b"idempotency-key",
        b"last-event-id",
        b"transfer-encoding",
        b"x-api-key",
        b"x-request-id",
        b"anthropic-version",
    }
)


def v1_error_body(path: str, message: str, status: int, code: str) -> dict[str, Any]:
    """Error envelope for `/v1` paths — Anthropic vs OpenAI grammar."""
    if path == "/v1/messages" or path.startswith("/v1/messages/"):
        return anthropic_error_body(message, status)
    return openai_error_body(message, status, code)


def bad_content_length_body(path: str) -> dict[str, Any]:
    if is_openai_path(path):
        return v1_error_body(path, "invalid content-length", 400, "bad_request")
    return {"detail": "invalid content-length", "code": "bad_request"}


def ambiguous_request_headers(request: Request) -> str | None:
    """Reject security-sensitive duplicates before any API collapses them."""
    seen: set[bytes] = set()
    for raw_name, _raw_value in request.scope.get("headers", []):
        name = bytes(raw_name).lower()
        singleton = name in _SINGLETON_REQUEST_HEADERS or name.startswith(b"x-fx1-")
        if singleton and name in seen:
            return f"duplicate {name.decode('latin-1')} header"
        seen.add(name)
    if b"content-length" in seen and b"transfer-encoding" in seen:
        return "content-length and transfer-encoding must not be combined"
    if b"authorization" in seen and b"x-api-key" in seen:
        return "send exactly one authentication header"
    return None


def request_header_error_body(path: str, message: str) -> dict[str, Any]:
    if is_openai_path(path):
        return v1_error_body(path, message, 400, "bad_request")
    return {"detail": message, "code": "bad_request"}


async def buffer_request_body(request: Request) -> tuple[int, bytes]:
    """Read the ASGI entity once without buffering beyond the body cap."""
    size = 0
    body = io.BytesIO()
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            request._body = b""  # noqa: SLF001 — no downstream consumer on refusal
            return size, b""
        body.write(chunk)
    buffered = body.getvalue()
    request._body = buffered  # noqa: SLF001 — preserve entity for downstream parsing
    return size, buffered


def too_large_body(path: str) -> dict[str, Any]:
    detail = f"body exceeds {MAX_BODY_BYTES}-byte cap"
    if is_openai_path(path):
        return v1_error_body(path, detail, 413, "too_large")
    return {"detail": detail, "code": "too_large"}


async def request_ingress_refusal(request: Request) -> JSONResponse | None:
    """Validate singleton/framing headers and cap the actual ASGI entity."""
    ambiguous = ambiguous_request_headers(request)
    if ambiguous is not None:
        return JSONResponse(
            status_code=400,
            content=request_header_error_body(request.url.path, ambiguous),
        )
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    declared = request.headers.get("content-length")
    length: int | None = None
    if declared is not None:
        if re.fullmatch(r"[0-9]+", declared) is None:
            return JSONResponse(
                status_code=400,
                content=bad_content_length_body(request.url.path),
            )
        length = int(declared)
        if length > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content=too_large_body(request.url.path))
    actual_length, _body = await buffer_request_body(request)
    if actual_length > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content=too_large_body(request.url.path))
    if length is not None and actual_length != length:
        return JSONResponse(
            status_code=400,
            content=request_header_error_body(
                request.url.path, "content-length does not match request body"
            ),
        )
    return None
