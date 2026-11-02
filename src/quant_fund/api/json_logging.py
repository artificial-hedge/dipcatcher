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


def redact_text(text: str) -> str:
    """Replace known secret values. Short values are left alone."""
    redacted = text
    for name in _SECRET_ENV_NAMES:
        secret = os.environ.get(name, "")
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
    sys.stderr.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    sys.stderr.flush()


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
        return json.dumps(payload, sort_keys=True, default=str)


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
