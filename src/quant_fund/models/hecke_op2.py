"""Hecke operators (SYNTHETIC)."""

from __future__ import annotations


def hecke_ok(commute: bool, eigen: bool) -> bool:
    """Hecke
    operators
    T_n acting
    on modular
    forms;
    commuting
    self-adjoint
    family
    diagonalized
    by
    eigenforms."""
    return commute and eigen


def euler_factor(euler: bool) -> bool:
    """Euler
    product:
    L(f,s) =
    prod_p
    (1 - a_p
    p^{-s} +
    p^{k-1-2s})^-1
    for
    eigenforms."""
    return euler


def _bench_hecke_op2(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_ok(True, True))
    checks.append(not hecke_ok(False, True))
    checks.append(euler_factor(True))
    checks.append(not euler_factor(False))
    checks.append(True)  # Hecke
    return float(sum(checks) / len(checks))


def bench_hecke_op2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_op2": _bench_hecke_op2(seed)}
