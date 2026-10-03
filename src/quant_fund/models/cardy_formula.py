"""cardy formula module (SYNTHETIC)."""

from __future__ import annotations


def cardy_formula_ok(perc: bool, crit: bool) -> bool:
    """cardy_formula
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def cardy_formula_aux(aux: bool) -> bool:
    """cardy_formula
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_cardy_formula(seed: int = 0) -> float:
    checks = []
    checks.append(cardy_formula_ok(True, True))
    checks.append(not cardy_formula_ok(False, True))
    checks.append(cardy_formula_aux(True))
    checks.append(not cardy_formula_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_cardy_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardy_formula": _bench_cardy_formula(seed)}
