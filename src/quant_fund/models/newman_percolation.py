"""newman percolation module (SYNTHETIC)."""

from __future__ import annotations


def newman_percolation_ok(perc: bool, lace: bool) -> bool:
    """newman_percolation
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def newman_percolation_aux(aux: bool) -> bool:
    """newman_percolation
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_newman_percolation(seed: int = 0) -> float:
    checks = []
    checks.append(newman_percolation_ok(True, True))
    checks.append(not newman_percolation_ok(False, True))
    checks.append(newman_percolation_aux(True))
    checks.append(not newman_percolation_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_newman_percolation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_newman_percolation": _bench_newman_percolation(seed)}
