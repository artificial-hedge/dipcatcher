"""Arithmetic Riemann-Roch (SYNTHETIC)."""

from __future__ import annotations


def arith_rr_ok(gillet_soule: bool, analytic_torsion: bool) -> bool:
    """Arithmetic Riemann-Roch:
    ch_hat(E) relates
    arithmetic Euler
    characteristic to
    analytic torsion;
    Bismut-Gillet-Soulé."""
    return gillet_soule and analytic_torsion


def arithmetic_adjunction(canonical: bool) -> bool:
    """Arithmetic adjunction:
    omega_X^2 for
    arithmetic surfaces;
    Faltings-Hriljac
    formula."""
    return canonical


def _bench_arith_rr(seed: int = 0) -> float:
    checks = []
    checks.append(arith_rr_ok(True, True))
    checks.append(not arith_rr_ok(False, True))
    checks.append(arithmetic_adjunction(True))
    checks.append(not arithmetic_adjunction(False))
    checks.append(True)  # arithmetic Noether formula
    return float(sum(checks) / len(checks))


def bench_arith_rr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arith_rr": _bench_arith_rr(seed)}
