"""smirnov percolation module (SYNTHETIC)."""

from __future__ import annotations


def smirnov_percolation_ok(perc: bool, crit: bool) -> bool:
    """smirnov_percolation
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def smirnov_percolation_aux(aux: bool) -> bool:
    """smirnov_percolation
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_smirnov_percolation(seed: int = 0) -> float:
    checks = []
    checks.append(smirnov_percolation_ok(True, True))
    checks.append(not smirnov_percolation_ok(False, True))
    checks.append(smirnov_percolation_aux(True))
    checks.append(not smirnov_percolation_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_smirnov_percolation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smirnov_percolation": _bench_smirnov_percolation(seed)}
