"""Redacted harness-client transport errors and the request deadline."""

from __future__ import annotations

from quant_fund.utils.net_timeout import require_timeout
from quant_fund.utils.secrets import redact_text


def timeout_s(value: float) -> float:
    return require_timeout(value, name="timeout_s")


def unreachable(url: str, reason: object) -> str:
    return redact_text(f"harness unreachable at {url}: {reason}")


def fault(url: str, exc: object) -> str:
    return redact_text(f"harness transport fault at {url}: {exc}")


def bad_base_url(base_url: str) -> str:
    return "base_url must be http(s)://host[:port], got " + repr(redact_text(base_url))
