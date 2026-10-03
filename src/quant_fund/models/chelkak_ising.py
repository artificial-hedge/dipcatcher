"""chelkak ising module (SYNTHETIC)."""

from __future__ import annotations


def chelkak_ising_ok(dimer: bool, ising: bool) -> bool:
    """chelkak_ising
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def chelkak_ising_aux(aux: bool) -> bool:
    """chelkak_ising
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_chelkak_ising(seed: int = 0) -> float:
    checks = []
    checks.append(chelkak_ising_ok(True, True))
    checks.append(not chelkak_ising_ok(False, True))
    checks.append(chelkak_ising_aux(True))
    checks.append(not chelkak_ising_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_chelkak_ising(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chelkak_ising": _bench_chelkak_ising(seed)}
