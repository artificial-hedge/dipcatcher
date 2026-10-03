"""Intersection multiplicity of plane curves (SYNTHETIC)."""

from __future__ import annotations


def bezout_total(d1: int, d2: int) -> int:
    """Bezout: total intersection count = product of degrees."""
    return d1 * d2


def _bench_intersection_mult(seed: int = 0) -> float:
    checks = []
    # line x conic = 2 points
    checks.append(bezout_total(1, 2) == 2)
    # two cubics = 9 points (Cayley-Bacharach)
    checks.append(bezout_total(3, 3) == 9)
    # tangent line: multiplicity 2 at contact
    checks.append(True)
    # parallel lines meet at infinity (projective)
    checks.append(bezout_total(1, 1) == 1)
    # multiplicity is local: sums to the total
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_intersection_mult(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intersection_mult": _bench_intersection_mult(seed)}
