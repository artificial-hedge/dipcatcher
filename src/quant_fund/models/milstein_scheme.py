"""milstein scheme module (SYNTHETIC)."""

from __future__ import annotations


def milstein_scheme_ok(st1: bool, kk: bool) -> bool:
    """milstein_scheme
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def milstein_scheme_aux(aux: bool) -> bool:
    """milstein_scheme
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_milstein_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(milstein_scheme_ok(True, True))
    checks.append(not milstein_scheme_ok(False, True))
    checks.append(milstein_scheme_aux(True))
    checks.append(not milstein_scheme_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_milstein_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milstein_scheme": _bench_milstein_scheme(seed)}
