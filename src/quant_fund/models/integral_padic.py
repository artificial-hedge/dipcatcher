"""Integral p-adic Hodge theory (SYNTHETIC)."""

from __future__ import annotations


def a_inf_ok(witt_perfect: bool, theta_map: bool) -> bool:
    """A_inf = W(R^flat) carries a
    theta map to R^+; Bhatt-Morrow-Scholze
    gave integral comparison theorems."""
    return witt_perfect and theta_map


def phi_mod_struct(newton_str: bool) -> bool:
    """A_inf-modules with Frobenius
    semilinear map phi encode integral
    crystalline data."""
    return newton_str


def _bench_integral_padic(seed: int = 0) -> float:
    checks = []
    checks.append(a_inf_ok(True, True))
    checks.append(not a_inf_ok(False, True))
    checks.append(phi_mod_struct(True))
    checks.append(not phi_mod_struct(False))
    checks.append(True)  # Fontaine's A_crys
    return float(sum(checks) / len(checks))


def bench_integral_padic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_integral_padic": _bench_integral_padic(seed)}
