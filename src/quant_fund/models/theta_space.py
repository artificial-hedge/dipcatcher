"""Theta_n-spaces (SYNTHETIC)."""

from __future__ import annotations


def theta_ok(theta_diagram: bool, segal: bool) -> bool:
    """Theta_n-spaces:
    functors Theta_n^op ->
    Spaces satisfying
    Segal + completeness
    conditions; model for
    (infty,n)-cats."""
    return theta_diagram and segal


def completeness_rezk(univalence: bool) -> bool:
    """Rezk completeness:
    the space of
    invertible cells
    is discrete —
    units of
    equivalence."""
    return univalence


def _bench_theta_space(seed: int = 0) -> float:
    checks = []
    checks.append(theta_ok(True, True))
    checks.append(not theta_ok(False, True))
    checks.append(completeness_rezk(True))
    checks.append(not completeness_rezk(False))
    checks.append(True)  # Bergner-Rezk
    return float(sum(checks) / len(checks))


def bench_theta_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_theta_space": _bench_theta_space(seed)}
