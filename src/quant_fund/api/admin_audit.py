"""Append-only audit lines for admin-key mutations.

One JSON object per line: time, route, key id, request id, outcome, and
kind. The key id is the SHA-256 hex of the presented admin key. The line
never contains the key. ``kill-file``, ``ledger``, and ``config`` are the
mutation kinds. A route that is not one of those still gets ``compute`` or
``unclassified`` so a new admin mutation is recorded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("quant_fund.api.audit")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_KINDS = frozenset({"kill-file", "ledger", "config", "compute", "unclassified"})
_ROUTE_KIND = {
    ("POST", "/backtest"): "ledger",
    ("POST", "/portfolio/optimize"): "compute",
}
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ACTOR: ContextVar[_Actor | None] = ContextVar("dipcatcher_admin_audit", default=None)


@dataclass(frozen=True)
class _Actor:
    route: str
    key_id: str
    request_id: str
    secrets: tuple[str, ...]
    kind: str
    log_env: str


def note_mutation(kind: str, outcome: str) -> None:
    """Record a mutation when an admin-key request is on the stack."""
    actor = _ACTOR.get()
    if actor is None:
        return
    _emit(
        actor,
        kind=_known(kind),
        outcome=outcome,
    )


@contextmanager
def admin_context(route: str, presented: str, log_env: str = "QUANT_API_AUDIT_LOG"):
    """Bind an admin actor. Tests and in-process callers use this."""
    actor = _actor_from_presented(route, presented, "unclassified", log_env, (presented,), None)
    token = _ACTOR.set(actor)
    try:
        yield
    finally:
        _ACTOR.reset(token)


async def audited_call(
    request: Any,
    call_next: Any,
    *,
    admin_current: str | None,
    admin_previous: str | None,
    read_current: str | None,
    read_previous: str | None,
    log_env: str,
) -> Any:
    """Run the route and append one line when the admin key authorized a mutation."""
    actor = _http_actor(
        request,
        admin_current,
        admin_previous,
        read_current,
        read_previous,
        log_env,
    )
    token = _ACTOR.set(actor)
    try:
        blocked = _blocked(actor)
        if blocked is not None:
            return blocked
        response = await call_next(request)
    finally:
        _ACTOR.reset(token)
    return _finish(actor, response)


def _http_actor(
    request: Any,
    admin_current: str | None,
    admin_previous: str | None,
    read_current: str | None,
    read_previous: str | None,
    log_env: str,
) -> _Actor | None:
    if request.method.upper() in _SAFE_METHODS:
        return None
    presented = request.headers.get("X-API-Key")
    from quant_fund.api.http_controls import admin_key_matches

    if not admin_key_matches(presented, admin_current, admin_previous):
        return None
    secrets = _secret_values(
        presented,
        admin_current,
        admin_previous,
        read_current,
        read_previous,
    )
    return _actor_from_presented(
        f"{request.method.upper()} {request.url.path}",
        presented or "",
        _route_kind(request.method, request.url.path),
        log_env,
        secrets,
        request.headers.get("X-Request-ID"),
    )


def _actor_from_presented(
    route: str,
    presented: str,
    kind: str,
    log_env: str,
    secrets: tuple[str, ...],
    header: str | None,
) -> _Actor:
    return _Actor(
        route=route,
        key_id=_key_id(presented),
        request_id=_request_id(header, secrets),
        secrets=secrets,
        kind=kind,
        log_env=log_env,
    )


def _blocked(actor: _Actor | None) -> Any:
    if actor is None or _log_ready(actor.log_env):
        return None
    _LOGGER.info("audit log unavailable")
    return _unavailable()


def _finish(actor: _Actor | None, response: Any) -> Any:
    if actor is None:
        return response
    if _emit(actor, kind=actor.kind, outcome=str(_status_code(response))):
        return response
    return _unavailable()


def _emit(actor: _Actor, *, kind: str, outcome: str) -> bool:
    record = {
        "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "route": _field(actor.route, actor.secrets),
        "key_id": _field(actor.key_id, actor.secrets),
        "request_id": _field(actor.request_id, actor.secrets),
        "outcome": _field(outcome, actor.secrets),
        "kind": _field(_known(kind), actor.secrets),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    if _contains_secret(line, actor.secrets):
        _LOGGER.info("audit line withheld")
        return _write(actor.log_env, '{"kind":"unclassified","outcome":"withheld"}\n')
    return _write(actor.log_env, line + "\n")


def _route_kind(method: str, path: str) -> str:
    return _ROUTE_KIND.get((method.upper(), path), "unclassified")


def _known(kind: str) -> str:
    if kind in _KINDS:
        return kind
    return "unclassified"


def _secret_values(*values: str | None) -> tuple[str, ...]:
    found: list[str] = []
    for value in values:
        if value:
            found.append(value)
    return tuple(found)


def _key_id(presented: str) -> str:
    try:
        raw = presented.encode("utf-8")
    except UnicodeEncodeError:
        raw = b"unencodable-api-key"
    return hashlib.sha256(raw).hexdigest()


def _request_id(header: str | None, secrets: tuple[str, ...]) -> str:
    if header and _REQUEST_ID.fullmatch(header) and not _contains_secret(header, secrets):
        return header
    return uuid.uuid4().hex


def _field(value: str, secrets: tuple[str, ...]) -> str:
    if _contains_secret(value, secrets):
        return "redacted"
    return value[:180]


def _contains_secret(value: str, secrets: tuple[str, ...]) -> bool:
    return any(bool(secret) and secret in value for secret in secrets)


def _status_code(response: Any) -> int:
    code = response.status_code
    if isinstance(code, int):
        return code
    return 0


def _log_ready(log_env: str) -> bool:
    raw = os.environ.get(log_env, "").strip()
    if not raw:
        return True
    try:
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError:
        return False
    os.close(fd)
    return True


def _write(log_env: str, line: str) -> bool:
    _LOGGER.info("%s", line.rstrip("\n"))
    raw = os.environ.get(log_env, "").strip()
    if not raw:
        return True
    try:
        _append_file(Path(raw), line)
    except (OSError, UnicodeEncodeError):
        return False
    return True


def _append_file(path: Path, line: str) -> None:
    payload = line.encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _unavailable() -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": "audit log unavailable"})
