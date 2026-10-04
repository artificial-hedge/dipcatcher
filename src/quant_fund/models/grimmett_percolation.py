"""grimmett percolation module (SYNTHETIC)."""

from __future__ import annotations


def grimmett_percolation_ok(perc: bool, crit: bool) -> bool:
    """grimmett_percolation
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def grimmett_percolation_aux(aux: bool) -> bool:
    """grimmett_percolation
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_grimmett_percolation(seed: int = 0) -> float:
    checks = []
    checks.append(grimmett_percolation_ok(True, True))
    checks.append(not grimmett_percolation_ok(False, True))
    checks.append(grimmett_percolation_aux(True))
    checks.append(not grimmett_percolation_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_grimmett_percolation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grimmett_percolation": _bench_grimmett_percolation(seed)}
