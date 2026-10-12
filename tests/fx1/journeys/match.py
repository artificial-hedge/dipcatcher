"""Shared expect/forbid matching and hang-guard bounds."""

from __future__ import annotations

import re
import time
from typing import Any

from tests.fx1.journeys.textutil import snippet, substitute

MIN_HANG_S = 60.0
STARTUP_HANG_S = 90.0


def hang_seconds(step: dict[str, Any], default: float = 75.0) -> float:
    """Per-step hang guard. Never below 60s."""
    raw = step.get("hang_s", default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    if value < MIN_HANG_S:
        return MIN_HANG_S
    return value


def match_output(step: dict[str, Any], region: str, *, stub_url: str, work: str) -> str | None:
    """None when every expect hits and no forbid hits."""
    try:
        return _match(step, region, stub_url=stub_url, work=work)
    except re.error as exc:
        return f"bad journey regex: {exc}"


def wait_requests(proc: Any, stub: Any, seen: int, hang_s: float) -> bool:
    """True once the stub has recorded a new request. Drains a pty if given."""
    deadline = time.monotonic() + hang_s
    while time.monotonic() < deadline:
        if _seen(stub, seen):
            return True
        if proc is None:
            time.sleep(0.05)
            continue
        if not proc.pump(0.1) and not _seen(stub, seen):
            return False
    return _seen(stub, seen)


def _seen(stub: Any, seen: int) -> bool:
    return stub is not None and len(stub.requests) > seen


def _match(step: dict[str, Any], region: str, *, stub_url: str, work: str) -> str | None:
    for pattern in step.get("expect", []):
        if _search(pattern, region, stub_url=stub_url, work=work) is None:
            return f"expected /{pattern}/ not in output: {snippet(region)}"
    for pattern in step.get("forbid", []):
        if _search(pattern, region, stub_url=stub_url, work=work) is not None:
            return f"forbidden /{pattern}/ appeared: {snippet(region)}"
    return None


def _search(pattern: str, region: str, *, stub_url: str, work: str) -> re.Match[str] | None:
    compiled = re.compile(substitute(pattern, stub_url=stub_url, work=work), re.MULTILINE)
    return compiled.search(region)
