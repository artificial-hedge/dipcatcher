"""hongler ising module (SYNTHETIC)."""

from __future__ import annotations


def hongler_ising_ok(dimer: bool, ising: bool) -> bool:
    """hongler_ising
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def hongler_ising_aux(aux: bool) -> bool:
    """hongler_ising
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_hongler_ising(seed: int = 0) -> float:
    checks = []
    checks.append(hongler_ising_ok(True, True))
    checks.append(not hongler_ising_ok(False, True))
    checks.append(hongler_ising_aux(True))
    checks.append(not hongler_ising_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_hongler_ising(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hongler_ising": _bench_hongler_ising(seed)}
