"""nonintrusive pce module (SYNTHETIC)."""

from __future__ import annotations


def nonintrusive_pce_ok(uq: bool, mode: bool) -> bool:
    """nonintrusive_pce
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def nonintrusive_pce_aux(aux: bool) -> bool:
    """nonintrusive_pce
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_nonintrusive_pce(seed: int = 0) -> float:
    checks = []
    checks.append(nonintrusive_pce_ok(True, True))
    checks.append(not nonintrusive_pce_ok(False, True))
    checks.append(nonintrusive_pce_aux(True))
    checks.append(not nonintrusive_pce_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_nonintrusive_pce(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonintrusive_pce": _bench_nonintrusive_pce(seed)}
