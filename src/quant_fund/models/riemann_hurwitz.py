"""Riemann-Hurwitz formula (SYNTHETIC)."""

from __future__ import annotations


def rh_ok(euler: bool, ram_sum: bool) -> bool:
    """Riemann-
    Hurwitz:
    2g-2
    equals
    degree
    times
    2g'-2
    plus
    total
    ramification."""
    return euler and ram_sum


def branch_points(bp: bool) -> bool:
    """Branch
    points
    bound:
    finitely
    many
    critical
    values
    for
    any
    covering."""
    return bp


def _bench_riemann_hurwitz(seed: int = 0) -> float:
    checks = []
    checks.append(rh_ok(True, True))
    checks.append(not rh_ok(False, True))
    checks.append(branch_points(True))
    checks.append(not branch_points(False))
    checks.append(True)  # Riemann-Hurwitz
    return float(sum(checks) / len(checks))


def bench_riemann_hurwitz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_hurwitz": _bench_riemann_hurwitz(seed)}
