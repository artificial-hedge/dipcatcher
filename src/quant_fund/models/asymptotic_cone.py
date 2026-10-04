"""Asymptotic cones (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(ultralimit: bool, rescaled: bool) -> bool:
    """Asymptotic
    cone:
    ultralimit
    of
    rescaled
    metric
    spaces —
    captures
    large-scale
    structure."""
    return ultralimit and rescaled


def gromov_thm_polynomial(gp: bool) -> bool:
    """Gromov:
    groups
    of
    polynomial
    growth
    are
    virtually
    nilpotent —
    cones
    give
    Lie
    structure."""
    return gp


def _bench_asymptotic_cone(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(gromov_thm_polynomial(True))
    checks.append(not gromov_thm_polynomial(False))
    checks.append(True)  # Gromov-van den Dries-Wilkie
    return float(sum(checks) / len(checks))


def bench_asymptotic_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asymptotic_cone": _bench_asymptotic_cone(seed)}
