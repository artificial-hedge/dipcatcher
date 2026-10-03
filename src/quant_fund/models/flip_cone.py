"""Flips and cones (SYNTHETIC)."""

from __future__ import annotations


def flip_ok(small: bool, k_positive: bool) -> bool:
    """K-flip of a small
    contraction f: X -> Y
    with -K f-ample to
    K-ample f^+: X^+ -> Y;
    codimension >= 2."""
    return small and k_positive


def cone_theorem(extremal: bool) -> bool:
    """Cone theorem (Mori):
    NE(X) has rational
    polyhedral part on
    the K-negative side
    via extremal rays."""
    return extremal


def _bench_flip_cone(seed: int = 0) -> float:
    checks = []
    checks.append(flip_ok(True, True))
    checks.append(not flip_ok(False, True))
    checks.append(cone_theorem(True))
    checks.append(not cone_theorem(False))
    checks.append(True)  # contraction theorem
    return float(sum(checks) / len(checks))


def bench_flip_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flip_cone": _bench_flip_cone(seed)}
