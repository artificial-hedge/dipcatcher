"""Latches for a split brain, a dependency outage, and a tampered journal.

The paper loop sets these. Readiness and the Prometheus text read them.
This module does not import the paper or api packages.
"""

from __future__ import annotations

_SPLIT = False
_DEPENDENCY: str | None = None
_TAMPER: str | None = None
_DEPENDENCY_CODES = frozenset({"dns", "tls", "broker_down", "broker_slow"})
_TAMPER_CODES = frozenset({"truncated", "reordered", "edited"})


def note_split_brain() -> None:
    """Latch a lost or refused paper-order lease."""
    global _SPLIT
    _SPLIT = True


def split_brain() -> bool:
    return _SPLIT


def note_dependency(code: str) -> None:
    """Latch a failed paper dependency probe."""
    global _DEPENDENCY
    _DEPENDENCY = code if code in _DEPENDENCY_CODES else "broker_down"


def clear_dependency() -> None:
    """Drop the dependency latch after a probe succeeds."""
    global _DEPENDENCY
    _DEPENDENCY = None


def dependency_code() -> str | None:
    return _DEPENDENCY


def note_audit_tamper(code: str) -> None:
    """Latch a journal or audit chain that failed at startup."""
    global _TAMPER
    _TAMPER = code if code in _TAMPER_CODES else "edited"


def audit_tamper() -> str | None:
    return _TAMPER


def clear_order_guard() -> None:
    """Test helper. A new process starts clear."""
    global _SPLIT, _DEPENDENCY, _TAMPER
    _SPLIT = False
    _DEPENDENCY = None
    _TAMPER = None


def order_guard_block() -> str | None:
    """Stable readiness code, or ``None`` when these latches are clear."""
    if _SPLIT:
        return "split_brain"
    if _DEPENDENCY is not None:
        return f"dependency_{_DEPENDENCY}"
    if _TAMPER is not None:
        return f"audit_{_TAMPER}"
    return None
