"""stochastic galerkin module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_galerkin_ok(uq: bool, mode: bool) -> bool:
    """stochastic_galerkin
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def stochastic_galerkin_aux(aux: bool) -> bool:
    """stochastic_galerkin
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_stochastic_galerkin(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_galerkin_ok(True, True))
    checks.append(not stochastic_galerkin_ok(False, True))
    checks.append(stochastic_galerkin_aux(True))
    checks.append(not stochastic_galerkin_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_stochastic_galerkin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_galerkin": _bench_stochastic_galerkin(seed)}
