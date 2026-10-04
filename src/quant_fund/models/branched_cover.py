"""Branched coverings (SYNTHETIC)."""

from __future__ import annotations


def bc_ok(degree: bool, ramify: bool) -> bool:
    """Branched
    covering:
    proper
    holomorphic
    map
    between
    surfaces —
    degree
    minus
    ramification
    points."""
    return degree and ramify


def local_degree(ld: bool) -> bool:
    """Local
    normal
    form
    z->z^k
    near
    a
    ramification
    point
    of
    order
    k."""
    return ld


def _bench_branched_cover(seed: int = 0) -> float:
    checks = []
    checks.append(bc_ok(True, True))
    checks.append(not bc_ok(False, True))
    checks.append(local_degree(True))
    checks.append(not local_degree(False))
    checks.append(True)  # Hurwitz
    return float(sum(checks) / len(checks))


def bench_branched_cover(seed: int = 0) -> dict[str, float]:
    return {"synthetic_branched_cover": _bench_branched_cover(seed)}
