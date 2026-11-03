"""Append-only audit lines for admin-key mutations and sensitive reads.

One JSON object per line: time, route, key id, request id, outcome, kind,
and prev. The key id is the SHA-256 hex of the presented key. ``prev``
is the SHA-256 hex of the previous raw line. The line never contains the key.
``kill-file``, ``ledger``, ``config``, ``optimize``, and ``positions`` are
the known kinds. A route that is none of those gets ``unclassified``.
``GET`` and ``HEAD`` of a ledger artifact or a position route append outcome
``read`` when the reads flag is on, which is the default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("quant_fund.api.audit")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_KINDS = frozenset({"kill-file", "ledger", "config", "optimize", "positions", "unclassified"})
_ROUTE_KIND = {
    ("POST", "/backtest"): "ledger",
    ("POST", "/portfolio/optimize"): "optimize",
}
_POSITION_PATHS = frozenset({"/risk/portfolio", "/portfolio/target", "/portfolio/exposures"})
_READS_OFF = frozenset({"0", "off", "false", "no"})
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ACTOR_TOKEN = re.compile(r"[^A-Za-z0-9._-]")
_ACTOR: ContextVar[_Actor | None] = ContextVar("dipcatcher_admin_audit", default=None)
_GENESIS = hashlib.sha256(b"").hexdigest()
_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()
_DEFAULT_MAX_BYTES = 1_048_576
_ROTATE_RESERVE = 1024
_TAIL_BYTES = 65536


@dataclass(frozen=True)
class _Actor:
    route: str
    key_id: str
    request_id: str
    secrets: tuple[str, ...]
    kind: str
    log_env: str
    fixed_outcome: str | None = None


def audit_log_env_names() -> tuple[str, ...]:
    """Log paths, the byte caps that rotate them, and the sensitive-read flags."""
    return (
        "QUANT_API_AUDIT_LOG",
        "RESEARCH_API_AUDIT_LOG",
        "QUANT_API_AUDIT_LOG_MAX_BYTES",
        "RESEARCH_API_AUDIT_LOG_MAX_BYTES",
        "QUANT_API_AUDIT_READS",
        "RESEARCH_API_AUDIT_READS",
    )


def cli_identity() -> str:
    """Process actor ``cli:<user>@<host>`` with an ASCII token on each side."""
    return f"cli:{_actor_token(_posix_user())}@{_actor_token(_host_name())}"


def note_mutation(kind: str, outcome: str) -> None:
    """Record a mutation when an admin actor is already on the stack."""
    actor = _ACTOR.get()
    if actor is None:
        return
    _emit(actor, kind=_known(kind), outcome=outcome)


def note_process_mutation(kind: str, outcome: str) -> None:
    """Record a config, ledger, or kill-file write from this process.

    An admin actor already on the stack keeps its route and key id.
    Otherwise the route is ``cli:<user>@<host>``.
    """
    if _ACTOR.get() is not None:
        note_mutation(kind, outcome)
        return
    identity = cli_identity()
    actor = _actor_from_presented(identity, identity, _known(kind), "QUANT_API_AUDIT_LOG", (), None)
    _emit(actor, kind=actor.kind, outcome=outcome)


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


def denied_audit_response(
    request: Any,
    refusal: Any,
    *,
    admin_current: str | None,
    admin_previous: str | None,
    read_current: str | None,
    read_previous: str | None,
    log_env: str,
) -> Any:
    """Return ``refusal``, or 503 when a denial line cannot be stored.

    Safe methods and a missing key are unchanged. An unsafe method that
    presented a key records outcome ``denied`` and does not copy the key.
    """
    if request.method.upper() in _SAFE_METHODS:
        return refusal
    presented = request.headers.get("X-API-Key")
    if not presented:
        return refusal
    secrets = _secret_values(presented, admin_current, admin_previous, read_current, read_previous)
    actor = _actor_from_presented(
        f"{request.method.upper()} {request.url.path}",
        presented,
        _route_kind(request.method, request.url.path),
        log_env,
        secrets,
        request.headers.get("X-Request-ID"),
    )
    if _emit(actor, kind=actor.kind, outcome="denied"):
        return refusal
    _LOGGER.info("audit log unavailable")
    return _unavailable()


def _http_actor(
    request: Any,
    admin_current: str | None,
    admin_previous: str | None,
    read_current: str | None,
    read_previous: str | None,
    log_env: str,
) -> _Actor | None:
    if request.method.upper() in _SAFE_METHODS:
        return _sensitive_read_actor(
            request,
            admin_current,
            admin_previous,
            read_current,
            read_previous,
            log_env,
        )
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


def _sensitive_read_actor(
    request: Any,
    admin_current: str | None,
    admin_previous: str | None,
    read_current: str | None,
    read_previous: str | None,
    log_env: str,
) -> _Actor | None:
    """Actor for a ledger or position read. Other safe methods stay quiet.

    The request already passed ``credential_refusal``. This path does not
    compare keys again. ``dump_resolved`` remains outcome ``written``; there
    is no HTTP config-dump route.
    """
    if not _reads_enabled(log_env):
        return None
    kind = _read_kind(request.url.path)
    if kind is None:
        return None
    presented = request.headers.get("X-API-Key") or ""
    secrets = _secret_values(
        presented,
        admin_current,
        admin_previous,
        read_current,
        read_previous,
    )
    return _actor_from_presented(
        f"{request.method.upper()} {request.url.path}",
        presented,
        kind,
        log_env,
        secrets,
        request.headers.get("X-Request-ID"),
        "read",
    )


def _reads_enabled(log_env: str) -> bool:
    name = f"{log_env.removesuffix('_LOG')}_READS"
    raw = os.environ.get(name, "on").strip().lower()
    return raw not in _READS_OFF


def _read_kind(path: str) -> str | None:
    if path in _POSITION_PATHS:
        return "positions"
    marker = "/backtest/"
    if path.startswith(marker) and len(path) > len(marker):
        return "ledger"
    return None


def _actor_from_presented(
    route: str,
    presented: str,
    kind: str,
    log_env: str,
    secrets: tuple[str, ...],
    header: str | None,
    fixed_outcome: str | None = None,
) -> _Actor:
    return _Actor(
        route=route,
        key_id=_key_id(presented),
        request_id=_request_id(header, secrets),
        secrets=secrets,
        kind=kind,
        log_env=log_env,
        fixed_outcome=fixed_outcome,
    )


def _blocked(actor: _Actor | None) -> Any:
    if actor is None or _log_ready(actor.log_env):
        return None
    _LOGGER.info("audit log unavailable")
    return _unavailable()


def _finish(actor: _Actor | None, response: Any) -> Any:
    if actor is None:
        return response
    outcome = actor.fixed_outcome if actor.fixed_outcome else str(_status_code(response))
    if _emit(actor, kind=actor.kind, outcome=outcome):
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
    if _contains_secret(_dumps(record), actor.secrets):
        _LOGGER.info("audit line withheld")
        return _commit(actor.log_env, {"kind": "unclassified", "outcome": "withheld"}, ())
    return _commit(actor.log_env, record, actor.secrets)


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


def _actor_token(value: str) -> str:
    cleaned = _ACTOR_TOKEN.sub("_", value)[:64]
    return cleaned or "unknown"


def _posix_user() -> str:
    try:
        import getpass

        return getpass.getuser()
    except OSError:
        return "unknown"


def _host_name() -> str:
    try:
        import socket

        return socket.gethostname()
    except OSError:
        return "unknown"


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


def _dumps(record: dict[str, str]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _withheld(prev: str) -> dict[str, str]:
    return {"kind": "unclassified", "outcome": "withheld", "prev": prev}


def _commit(log_env: str, record: dict[str, str], secrets: tuple[str, ...]) -> bool:
    raw = os.environ.get(log_env, "").strip()
    if not raw:
        body = dict(record)
        body["prev"] = _GENESIS
        line = _dumps(body)
        if _contains_secret(line, secrets):
            line = _dumps(_withheld(_GENESIS))
        _LOGGER.info("%s", line)
        return True
    try:
        with _path_lock(raw):
            path = Path(raw)
            _rotate(path, _max_bytes(log_env))
            prev = _prev_hash(path)
            body = dict(record)
            body["prev"] = prev
            line = _dumps(body)
            if _contains_secret(line, secrets):
                line = _dumps(_withheld(prev))
            _LOGGER.info("%s", line)
            _append_file(path, line + "\n")
    except (OSError, UnicodeError):
        return False
    return True


def _path_lock(raw: str) -> threading.Lock:
    with _LOCK_GUARD:
        lock = _LOCKS.get(raw)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[raw] = lock
        return lock


def _max_bytes(log_env: str) -> int | None:
    raw = os.environ.get(f"{log_env}_MAX_BYTES", "").strip().lower()
    if not raw:
        return _DEFAULT_MAX_BYTES
    if raw in {"0", "off", "false", "no"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BYTES
    if value <= 0:
        return None
    return value


def _rotate(path: Path, max_bytes: int | None) -> None:
    if max_bytes is None or not path.is_file():
        return
    size = path.stat().st_size
    if size == 0 or size + _ROTATE_RESERVE <= max_bytes:
        return
    _shift(path)


def _shift(path: Path) -> None:
    generation = 1
    while path.with_name(f"{path.name}.{generation}").exists():
        generation += 1
    while generation > 1:
        source = path.with_name(f"{path.name}.{generation - 1}")
        target = path.with_name(f"{path.name}.{generation}")
        os.replace(source, target)
        generation -= 1
    os.replace(path, path.with_name(f"{path.name}.1"))


def _prev_hash(path: Path) -> str:
    last = _last_line(path)
    if last is None:
        last = _last_line(path.with_name(f"{path.name}.1"))
    if last is None:
        return _GENESIS
    return hashlib.sha256(last.encode("ascii")).hexdigest()


def _last_line(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _TAIL_BYTES))
        blob = handle.read()
    lines = blob.decode("ascii").splitlines()
    if not lines:
        return None
    return lines[-1]


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
