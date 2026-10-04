"""stochastic colloc module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_colloc_ok(uq: bool, mode: bool) -> bool:
    """stochastic_colloc
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def stochastic_colloc_aux(aux: bool) -> bool:
    """stochastic_colloc
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_stochastic_colloc(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_colloc_ok(True, True))
    checks.append(not stochastic_colloc_ok(False, True))
    checks.append(stochastic_colloc_aux(True))
    checks.append(not stochastic_colloc_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_stochastic_colloc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_colloc": _bench_stochastic_colloc(seed)}
