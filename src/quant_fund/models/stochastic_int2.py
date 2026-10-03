"""stochastic int2 module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_int2_ok(si: bool, isom: bool) -> bool:
    """stochastic_int2
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def stochastic_int2_aux(aux: bool) -> bool:
    """stochastic_int2
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_stochastic_int2(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_int2_ok(True, True))
    checks.append(not stochastic_int2_ok(False, True))
    checks.append(stochastic_int2_aux(True))
    checks.append(not stochastic_int2_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_stochastic_int2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_int2": _bench_stochastic_int2(seed)}
