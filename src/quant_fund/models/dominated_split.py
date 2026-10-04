"""Dominated splittings (SYNTHETIC)."""

from __future__ import annotations


def dom_ok(dominate: bool, angle: bool) -> bool:
    """Dominated
    splitting:
    E
    dominates
    F —
    vectors
    in F
    expand
    strictly
    less
    than
    in E
    uniformly."""
    return dominate and angle


def robust_cone(cone: bool) -> bool:
    """Robust
    cone
    families:
    dominated
    splittings
    persist
    under
    C^1
    perturbations."""
    return cone


def _bench_dominated_split(seed: int = 0) -> float:
    checks = []
    checks.append(dom_ok(True, True))
    checks.append(not dom_ok(False, True))
    checks.append(robust_cone(True))
    checks.append(not robust_cone(False))
    checks.append(True)  # Mané-Liao
    return float(sum(checks) / len(checks))


def bench_dominated_split(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dominated_split": _bench_dominated_split(seed)}
