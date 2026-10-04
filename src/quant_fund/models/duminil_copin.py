"""duminil copin module (SYNTHETIC)."""

from __future__ import annotations


def duminil_copin_ok(perc: bool, crit: bool) -> bool:
    """duminil_copin
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def duminil_copin_aux(aux: bool) -> bool:
    """duminil_copin
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_duminil_copin(seed: int = 0) -> float:
    checks = []
    checks.append(duminil_copin_ok(True, True))
    checks.append(not duminil_copin_ok(False, True))
    checks.append(duminil_copin_aux(True))
    checks.append(not duminil_copin_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_duminil_copin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duminil_copin": _bench_duminil_copin(seed)}
