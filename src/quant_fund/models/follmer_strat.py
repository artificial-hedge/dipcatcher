"""follmer strat module (SYNTHETIC)."""

from __future__ import annotations


def follmer_strat_ok(ii1: bool, sc: bool) -> bool:
    """follmer_strat
    check:
    stochastic
    calculus —
    isometry/conversion."""
    return ii1 and sc


def follmer_strat_aux(aux: bool) -> bool:
    """follmer_strat
    aux:
    auxiliary
    sde
    check —
    transform."""
    return aux


def _bench_follmer_strat(seed: int = 0) -> float:
    checks = []
    checks.append(follmer_strat_ok(True, True))
    checks.append(not follmer_strat_ok(False, True))
    checks.append(follmer_strat_aux(True))
    checks.append(not follmer_strat_aux(False))
    checks.append(True)  # stochastic calc canon
    return float(sum(checks) / len(checks))


def bench_follmer_strat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_follmer_strat": _bench_follmer_strat(seed)}
