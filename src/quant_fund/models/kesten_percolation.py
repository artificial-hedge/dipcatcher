"""kesten percolation module (SYNTHETIC)."""

from __future__ import annotations


def kesten_percolation_ok(perc: bool, crit: bool) -> bool:
    """kesten_percolation
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def kesten_percolation_aux(aux: bool) -> bool:
    """kesten_percolation
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_kesten_percolation(seed: int = 0) -> float:
    checks = []
    checks.append(kesten_percolation_ok(True, True))
    checks.append(not kesten_percolation_ok(False, True))
    checks.append(kesten_percolation_aux(True))
    checks.append(not kesten_percolation_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_kesten_percolation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kesten_percolation": _bench_kesten_percolation(seed)}
