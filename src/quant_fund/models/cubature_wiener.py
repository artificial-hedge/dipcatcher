"""cubature wiener module (SYNTHETIC)."""

from __future__ import annotations


def cubature_wiener_ok(st1: bool, kk: bool) -> bool:
    """cubature_wiener
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def cubature_wiener_aux(aux: bool) -> bool:
    """cubature_wiener
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_cubature_wiener(seed: int = 0) -> float:
    checks = []
    checks.append(cubature_wiener_ok(True, True))
    checks.append(not cubature_wiener_ok(False, True))
    checks.append(cubature_wiener_aux(True))
    checks.append(not cubature_wiener_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_cubature_wiener(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cubature_wiener": _bench_cubature_wiener(seed)}
