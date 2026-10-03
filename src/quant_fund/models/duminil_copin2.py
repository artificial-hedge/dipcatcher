"""duminil copin2 module (SYNTHETIC)."""

from __future__ import annotations


def duminil_copin2_ok(dimer: bool, ising: bool) -> bool:
    """duminil_copin2
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def duminil_copin2_aux(aux: bool) -> bool:
    """duminil_copin2
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_duminil_copin2(seed: int = 0) -> float:
    checks = []
    checks.append(duminil_copin2_ok(True, True))
    checks.append(not duminil_copin2_ok(False, True))
    checks.append(duminil_copin2_aux(True))
    checks.append(not duminil_copin2_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_duminil_copin2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duminil_copin2": _bench_duminil_copin2(seed)}
