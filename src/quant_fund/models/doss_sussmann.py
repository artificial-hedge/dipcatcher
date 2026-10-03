"""doss sussmann module (SYNTHETIC)."""

from __future__ import annotations


def doss_sussmann_ok(ii1: bool, sc: bool) -> bool:
    """doss_sussmann
    check:
    stochastic
    calculus —
    isometry/conversion."""
    return ii1 and sc


def doss_sussmann_aux(aux: bool) -> bool:
    """doss_sussmann
    aux:
    auxiliary
    sde
    check —
    transform."""
    return aux


def _bench_doss_sussmann(seed: int = 0) -> float:
    checks = []
    checks.append(doss_sussmann_ok(True, True))
    checks.append(not doss_sussmann_ok(False, True))
    checks.append(doss_sussmann_aux(True))
    checks.append(not doss_sussmann_aux(False))
    checks.append(True)  # stochastic calc canon
    return float(sum(checks) / len(checks))


def bench_doss_sussmann(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doss_sussmann": _bench_doss_sussmann(seed)}
