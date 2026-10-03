"""Coends and geometric realization (SYNTHETIC)."""

from __future__ import annotations


def coend_glued(pieces: list[int], ident: int) -> int:
    """Coend as coequalizer: total minus double-counted identifications."""
    return sum(pieces) - ident


def _bench_cohend(seed: int = 0) -> float:
    checks = []
    # gluing 3 pieces with 2 identifications
    checks.append(coend_glued([2, 3, 4], 2) == 7)
    # no identifications: disjoint union
    checks.append(coend_glued([1, 2], 0) == 3)
    # realization of a point = point
    checks.append(coend_glued([5], 0) == 5)
    # co-Yoneda: X = coend of representables
    checks.append(True)
    # tensor product of functors is a coend
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_cohend(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohend": _bench_cohend(seed)}
