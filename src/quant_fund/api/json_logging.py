"""JSON lines on stderr for the service process. Values of known secret
environment variables are replaced before the line is written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_SECRET_ENV_NAMES = (
    "AGENT_GW_TOKEN",
    "BEA_API_KEY",
    "DATASOURCE_API_KEY",
    "DIPCATCHER_SIGSTORE_ID_TOKEN",
    "FRED_API_KEY",
    "FX1_API_KEY",
    "FX1_BYOK_API_KEY",
    "FX1_LOCAL_API_KEY",
    "FX1_SIGNING_KEY",
    "GATE_QUORUM_KEYS",
    "GATE_SIGNING_KEY",
    "GATE_SIGNING_KEY_NEW",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "PROOFCORE_SIGNING_KEY",
    "QUANT_API_KEY",
    "QUANT_VENDOR_API_KEY",
    "RESEARCH_API_KEY",
    "VENDOR_HTTP_API_KEY",
    "WITNESS_SIGNING_KEY",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)
_MIN_SECRET_LEN = 8
_REQUEST_ID: ContextVar[str | None] = ContextVar("dipcatcher_request_id", default=None)


def bind_request_id(value: str) -> Token[str | None]:
    """Remember ``value`` for log lines written on this task."""
    return _REQUEST_ID.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request id that was current before :func:`bind_request_id`."""
    _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()


def _secret_values() -> tuple[str, ...]:
    """Env values and both lines of a ``NAME_FILE``. Empty and short values drop out."""
    from quant_fund.utils.secret_tokens import read_secret_tokens

    found: list[str] = []
    for name in _SECRET_ENV_NAMES:
        path = os.environ.get(f"{name}_FILE", "").strip()
        if path:
            found.extend(read_secret_tokens(path) or ())
            continue
        plain = os.environ.get(name, "")
        if plain:
            found.append(plain)
    return tuple(found)


def redact_text(text: str) -> str:
    """Replace known secret values. Short values are left alone."""
    redacted = text
    for secret in _secret_values():
        if len(secret) >= _MIN_SECRET_LEN and secret in redacted:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def emit_service_event(event: str, **fields: Any) -> None:
    """Write one JSON object to stderr. ``fields`` must already be public."""
    record: dict[str, object] = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": "info",
        "event": event,
        "service": "dipcatcher",
    }
    for key, value in fields.items():
        if isinstance(value, str):
            record[str(key)] = redact_text(value)
        else:
            record[str(key)] = value
    request_id = current_request_id()
    if request_id and "request_id" not in record:
        record["request_id"] = request_id
    encoded = redact_text(json.dumps(record, sort_keys=True, default=str))
    sys.stderr.write(encoded + "\n")
    sys.stderr.flush()
    append_audit_line(encoded)


class JsonLineFormatter(logging.Formatter):
    """Uvicorn formatter. One JSON object per record, secrets removed."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname.lower(),
            "event": "log",
            "logger": record.name,
            "message": redact_text(record.getMessage()),
            "service": "dipcatcher",
        }
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        return redact_text(json.dumps(payload, sort_keys=True, default=str))


_AUDIT_PATH = "DIPCATCHER_AUDIT_LOG_PATH"
_AUDIT_MAX_BYTES = "DIPCATCHER_AUDIT_LOG_MAX_BYTES"
_AUDIT_MAX_FILES = "DIPCATCHER_AUDIT_LOG_MAX_FILES"
_DEFAULT_AUDIT_BYTES = 10 * 1024 * 1024
_DEFAULT_AUDIT_FILES = 5


def _positive_int(name: str, default: int) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdecimal():
        return None
    value = int(raw)
    if value < 1:
        return None
    return value


def _audit_limits() -> tuple[int, int] | None:
    max_bytes = _positive_int(_AUDIT_MAX_BYTES, _DEFAULT_AUDIT_BYTES)
    max_files = _positive_int(_AUDIT_MAX_FILES, _DEFAULT_AUDIT_FILES)
    if max_bytes is None or max_files is None:
        return None
    return max_bytes, max_files


def _shift_numbered(path: str, max_files: int) -> None:
    oldest = f"{path}.{max_files}"
    if os.path.islink(path) or os.path.islink(oldest):
        raise OSError("symlink")
    if os.path.lexists(oldest):
        os.unlink(oldest)
    index = max_files - 1
    while index >= 1:
        src = f"{path}.{index}"
        dest = f"{path}.{index + 1}"
        if os.path.islink(src) or os.path.islink(dest):
            raise OSError("symlink")
        if os.path.lexists(src):
            os.replace(src, dest)
        index -= 1
    os.replace(path, f"{path}.1")


def _open_audit(path: str, payload: bytes, limits: tuple[int, int]) -> int:
    max_bytes, max_files = limits
    if os.path.isfile(path) and os.path.getsize(path) + len(payload) > max_bytes:
        _shift_numbered(path, max_files)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)


def append_audit_line(line: str) -> None:
    """Append one already-redacted line when an absolute audit path is set.

    The service logger stays on stderr. This file is the audit stream. A
    symlink is ignored. When the next line would pass the size cap, the
    current file is renamed to ``path.1`` and older copies shift up.
    ``scripts/log_retain.py`` applies the age cap.
    """
    raw = os.environ.get(_AUDIT_PATH, "").strip()
    if not raw or not os.path.isabs(raw) or os.path.islink(raw):
        return
    limits = _audit_limits()
    if limits is None:
        return
    payload = (line + "\n").encode("utf-8")
    try:
        descriptor = _open_audit(raw, payload, limits)
    except OSError:
        return
    try:
        os.write(descriptor, payload)
    except OSError:
        return
    finally:
        os.close(descriptor)


def json_log_config(level: str) -> dict[str, Any]:
    """dictConfig mapping uvicorn can consume. Logs go to stderr only."""
    upper = level.upper()
    handler: dict[str, str] = {
        "class": "logging.StreamHandler",
        "formatter": "json",
        "stream": "ext://sys.stderr",
    }
    logger: dict[str, object] = {"handlers": ["stderr"], "level": upper, "propagate": False}
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": "quant_fund.api.json_logging.JsonLineFormatter"}},
        "handlers": {"stderr": handler},
        "root": {"handlers": ["stderr"], "level": upper},
        "loggers": {
            "uvicorn": dict(logger),
            "uvicorn.error": dict(logger),
            "uvicorn.access": dict(logger),
        },
    }
