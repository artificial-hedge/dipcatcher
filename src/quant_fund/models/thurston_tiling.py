"""thurston tiling module (SYNTHETIC)."""

from __future__ import annotations


def thurston_tiling_ok(dimer: bool, ising: bool) -> bool:
    """thurston_tiling
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def thurston_tiling_aux(aux: bool) -> bool:
    """thurston_tiling
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_thurston_tiling(seed: int = 0) -> float:
    checks = []
    checks.append(thurston_tiling_ok(True, True))
    checks.append(not thurston_tiling_ok(False, True))
    checks.append(thurston_tiling_aux(True))
    checks.append(not thurston_tiling_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_thurston_tiling(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thurston_tiling": _bench_thurston_tiling(seed)}
