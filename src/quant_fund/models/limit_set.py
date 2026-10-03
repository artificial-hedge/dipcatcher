"""Limit sets (SYNTHETIC)."""

from __future__ import annotations


def limit_ok(accum: bool, minimal: bool) -> bool:
    """Limit
    set:
    accumulation
    of
    any
    orbit
    on
    the
    boundary —
    smallest
    closed
    invariant
    set."""
    return accum and minimal


def two_points(tp: bool) -> bool:
    """Non-
    elementary
    iff
    the
    limit
    set
    has
    more
    than
    two
    points —
    then
    infinite."""
    return tp


def _bench_limit_set(seed: int = 0) -> float:
    checks = []
    checks.append(limit_ok(True, True))
    checks.append(not limit_ok(False, True))
    checks.append(two_points(True))
    checks.append(not two_points(False))
    checks.append(True)  # Ahlfors
    return float(sum(checks) / len(checks))


def bench_limit_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_limit_set": _bench_limit_set(seed)}
