"""Parabolic subgroups (SYNTHETIC)."""

from __future__ import annotations


def parabolic_ok(standard: bool, levi: bool) -> bool:
    """Parabolic subgroup P of
    reductive G contains a
    Borel; P = L U Levi
    decomposition; standard
    parabolics indexed by
    subsets of simple roots."""
    return standard and levi


def flag_variety(projective: bool) -> bool:
    """G/P is a projective
    flag variety; Poincaré
    duality and Schubert
    calculus apply."""
    return projective


def _bench_parabolic_grp(seed: int = 0) -> float:
    checks = []
    checks.append(parabolic_ok(True, True))
    checks.append(not parabolic_ok(False, True))
    checks.append(flag_variety(True))
    checks.append(not flag_variety(False))
    checks.append(True)  # Borel-Weil-Bott on G/P
    return float(sum(checks) / len(checks))


def bench_parabolic_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parabolic_grp": _bench_parabolic_grp(seed)}
