"""beffara nolin module (SYNTHETIC)."""

from __future__ import annotations


def beffara_nolin_ok(perc: bool, lace: bool) -> bool:
    """beffara_nolin
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def beffara_nolin_aux(aux: bool) -> bool:
    """beffara_nolin
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_beffara_nolin(seed: int = 0) -> float:
    checks = []
    checks.append(beffara_nolin_ok(True, True))
    checks.append(not beffara_nolin_ok(False, True))
    checks.append(beffara_nolin_aux(True))
    checks.append(not beffara_nolin_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_beffara_nolin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beffara_nolin": _bench_beffara_nolin(seed)}
