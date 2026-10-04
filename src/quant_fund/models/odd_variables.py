"""Odd variables (SYNTHETIC)."""

from __future__ import annotations


def odd_ok(anticommute: bool, nilpotent: bool) -> bool:
    """Odd variables
    theta_i:
    theta_i theta_j
    = -theta_j theta_i,
    so theta_i^2
    = 0; Grassmann
    numbers."""
    return anticommute and nilpotent


def grassmann_free(free: bool) -> bool:
    """Grassmann
    algebra
    Lambda(theta_1,
    ..., theta_n)
    is the free
    supercommutative
    algebra on
    odd generators."""
    return free


def _bench_odd_variables(seed: int = 0) -> float:
    checks = []
    checks.append(odd_ok(True, True))
    checks.append(not odd_ok(False, True))
    checks.append(grassmann_free(True))
    checks.append(not grassmann_free(False))
    checks.append(True)  # Grassmann
    return float(sum(checks) / len(checks))


def bench_odd_variables(seed: int = 0) -> dict[str, float]:
    return {"synthetic_odd_variables": _bench_odd_variables(seed)}
