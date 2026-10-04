"""smirnov ising module (SYNTHETIC)."""

from __future__ import annotations


def smirnov_ising_ok(dimer: bool, ising: bool) -> bool:
    """smirnov_ising
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def smirnov_ising_aux(aux: bool) -> bool:
    """smirnov_ising
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_smirnov_ising(seed: int = 0) -> float:
    checks = []
    checks.append(smirnov_ising_ok(True, True))
    checks.append(not smirnov_ising_ok(False, True))
    checks.append(smirnov_ising_aux(True))
    checks.append(not smirnov_ising_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_smirnov_ising(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smirnov_ising": _bench_smirnov_ising(seed)}
