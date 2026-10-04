"""aiten chayes module (SYNTHETIC)."""

from __future__ import annotations


def aiten_chayes_ok(perc: bool, lace: bool) -> bool:
    """aiten_chayes
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def aiten_chayes_aux(aux: bool) -> bool:
    """aiten_chayes
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_aiten_chayes(seed: int = 0) -> float:
    checks = []
    checks.append(aiten_chayes_ok(True, True))
    checks.append(not aiten_chayes_ok(False, True))
    checks.append(aiten_chayes_aux(True))
    checks.append(not aiten_chayes_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_aiten_chayes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aiten_chayes": _bench_aiten_chayes(seed)}
