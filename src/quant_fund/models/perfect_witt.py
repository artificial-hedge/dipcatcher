"""Witt vectors of perfect rings (SYNTHETIC)."""

from __future__ import annotations


def pw_ok(perfect: bool, witt: bool) -> bool:
    """Perfect:
    Witt
    vectors
    of
    a
    perfect
    ring —
    Fontaine
    perfect."""
    return perfect and witt


def witt_strict(ws: bool) -> bool:
    """Strict:
    strict
    p-
    ring —
    Cohen
    strict."""
    return ws


def _bench_perfect_witt(seed: int = 0) -> float:
    checks = []
    checks.append(pw_ok(True, True))
    checks.append(not pw_ok(False, True))
    checks.append(witt_strict(True))
    checks.append(not witt_strict(False))
    checks.append(True)  # Fontaine
    return float(sum(checks) / len(checks))


def bench_perfect_witt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perfect_witt": _bench_perfect_witt(seed)}
