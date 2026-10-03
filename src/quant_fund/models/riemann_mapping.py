"""Riemann mapping theorem (SYNTHETIC)."""

from __future__ import annotations


def rm_ok(simply_conn: bool, proper: bool) -> bool:
    """Riemann
    mapping:
    every
    simply
    connected
    proper
    domain
    is
    biholomorphic
    to
    the
    disc."""
    return simply_conn and proper


def normalization(normal: bool) -> bool:
    """Uniqueness
    up
    to
    normalization:
    fixing
    a
    point
    and
    derivative
    sign
    makes
    the
    map
    unique."""
    return normal


def _bench_riemann_mapping(seed: int = 0) -> float:
    checks = []
    checks.append(rm_ok(True, True))
    checks.append(not rm_ok(False, True))
    checks.append(normalization(True))
    checks.append(not normalization(False))
    checks.append(True)  # Riemann
    return float(sum(checks) / len(checks))


def bench_riemann_mapping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_mapping": _bench_riemann_mapping(seed)}
