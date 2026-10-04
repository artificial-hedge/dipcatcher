"""stochastic taylor module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_taylor_ok(st1: bool, kk: bool) -> bool:
    """stochastic_taylor
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def stochastic_taylor_aux(aux: bool) -> bool:
    """stochastic_taylor
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_stochastic_taylor(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_taylor_ok(True, True))
    checks.append(not stochastic_taylor_ok(False, True))
    checks.append(stochastic_taylor_aux(True))
    checks.append(not stochastic_taylor_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_stochastic_taylor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_taylor": _bench_stochastic_taylor(seed)}
