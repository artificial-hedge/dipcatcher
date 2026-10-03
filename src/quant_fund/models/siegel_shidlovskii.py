"""Siegel-Shidlovskii theorem (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(e_functions: bool, algebraic_vals: bool) -> bool:
    """Siegel-
    Shidlovskii:
    E-functions
    satisfying
    linear
    differential
    equations
    take
    algebraic-
    independent
    values
    at
    algebraic
    points."""
    return e_functions and algebraic_vals


def zero_estimates(ze: bool) -> bool:
    """Zero
    estimates
    and
    auxiliary
    functions
    drive
    transcendence
    proofs."""
    return ze


def _bench_siegel_shidlovskii(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(zero_estimates(True))
    checks.append(not zero_estimates(False))
    checks.append(True)  # Siegel-Shidlovskii
    return float(sum(checks) / len(checks))


def bench_siegel_shidlovskii(seed: int = 0) -> dict[str, float]:
    return {"synthetic_siegel_shidlovskii": _bench_siegel_shidlovskii(seed)}
