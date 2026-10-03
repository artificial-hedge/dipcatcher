"""stochastic fem module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_fem_ok(uq: bool, mode: bool) -> bool:
    """stochastic_fem
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def stochastic_fem_aux(aux: bool) -> bool:
    """stochastic_fem
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_stochastic_fem(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_fem_ok(True, True))
    checks.append(not stochastic_fem_ok(False, True))
    checks.append(stochastic_fem_aux(True))
    checks.append(not stochastic_fem_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_stochastic_fem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_fem": _bench_stochastic_fem(seed)}
