"""Shared OpenAI request option validation and rejection taxonomy."""

from __future__ import annotations

import math
from typing import Any

from fx1.serve.receipt_store import SHA256_HEX


class OpenAICompatError(ValueError):
    """A rejection carrying the wire status and OpenAI error code.

    The SDK surfaces this as a ``ValueError``; the API maps it to its
    response envelope without changing the validation verdict.
    """

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def history_tool_call_shape(raw: Any, i: int, j: int) -> dict[str, Any]:
    """Validate one assistant tool-call replay entry before forwarding it."""
    fn = raw.get("function") if isinstance(raw, dict) else None
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("id"), str)
        or raw.get("type") != "function"
        or not isinstance(fn, dict)
        or not isinstance(fn.get("name"), str)
        or not isinstance(args, str)
    ):
        raise OpenAICompatError(
            f"messages[{i}].tool_calls[{j}]: needs "
            "{id, type: 'function', function: {name, arguments}}"
        )
    return raw


def resolve_timeout(ext_timeout: float | None, hdrs: dict[str, str]) -> float | None:
    """Apply the body/header timeout precedence and require a finite bound."""
    if ext_timeout is not None:
        return ext_timeout
    raw = hdrs.get("x-fx1-timeout")
    if raw is None:
        return None
    try:
        val = float(raw)
    except ValueError as exc:
        raise OpenAICompatError("X-Fx1-Timeout must be seconds as a number") from exc
    if not math.isfinite(val) or val <= 0.0 or val > 3600.0:
        raise OpenAICompatError("X-Fx1-Timeout must be in (0, 3600] seconds")
    return val


def resolve_receipt_hashes(ext_hashes: list[str] | None, hdrs: dict[str, str]) -> list[str] | None:
    """Validate optional receipt citations from body or HTTP header."""
    if ext_hashes:
        if any(SHA256_HEX.fullmatch(h) is None for h in ext_hashes):
            raise OpenAICompatError("fx1.receipt_hashes entries must be sha256 digests")
        return list(ext_hashes)
    raw = hdrs.get("x-fx1-receipt-hashes")
    if raw is None:
        return None
    out = [h.strip() for h in raw.split(",") if h.strip()]
    if any(SHA256_HEX.fullmatch(h) is None for h in out):
        raise OpenAICompatError("X-Fx1-Receipt-Hashes must be comma-separated sha256 digests")
    return out or None
