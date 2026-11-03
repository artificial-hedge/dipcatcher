"""One JSON line per HTTP request, with the id the proxy forwarded.

A request id is accepted only when it is a single token of letters, digits,
``.``, ``_``, or ``-``. Anything else is replaced before it is logged or
echoed. The line does not include headers, the query string, or the body.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from quant_fund.api.json_logging import bind_request_id, emit_service_event, reset_request_id
from quant_fund.observe.metrics import record_http_request

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SERVICES = frozenset({"lab", "research"})


def accept_request_id(value: str) -> str:
    """Return ``value`` when it is a safe token, otherwise a new id."""
    text = value.strip()
    if _ID.fullmatch(text):
        return text
    return uuid.uuid4().hex


class RequestLogMiddleware:
    """Outermost ASGI middleware. Echoes ``X-Request-Id`` and writes one log line."""

    def __init__(self, app: Any, service: str) -> None:
        self.app = app
        self.service = service if service in _SERVICES else "lab"

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = accept_request_id(_header(scope, b"x-request-id"))
        token = bind_request_id(request_id)
        status = 500
        started = time.perf_counter()

        async def wrap(message: dict[str, Any]) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 500))
                _ensure_request_header(message, request_id)
            await send(message)

        try:
            await self.app(scope, receive, wrap)
        finally:
            seconds = time.perf_counter() - started
            record_http_request(self.service, status, seconds)
            emit_service_event(
                "request",
                api=self.service,
                duration_ms=_duration_ms(seconds),
                method=str(scope.get("method", "")),
                path=str(scope.get("path", "")),
                status=status,
            )
            reset_request_id(token)


def _header(scope: dict[str, Any], name: bytes) -> str:
    for key, value in scope.get("headers") or []:
        if bytes(key).lower() == name:
            try:
                return bytes(value).decode("ascii")
            except UnicodeDecodeError:
                return ""
    return ""


def _ensure_request_header(message: dict[str, Any], request_id: str) -> None:
    headers = message.setdefault("headers", [])
    for key, _value in headers:
        if bytes(key).lower() == b"x-request-id":
            return
    headers.append((b"x-request-id", request_id.encode("ascii")))


def _duration_ms(seconds: float) -> float:
    if seconds < 0:
        return 0.0
    return round(seconds * 1000.0, 3)
