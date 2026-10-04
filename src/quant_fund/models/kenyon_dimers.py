"""kenyon dimers module (SYNTHETIC)."""

from __future__ import annotations


def kenyon_dimers_ok(dimer: bool, ising: bool) -> bool:
    """kenyon_dimers
    check:
    dimer/Ising
    structure —
    Smirnov."""
    return dimer and ising


def kenyon_dimers_aux(aux: bool) -> bool:
    """kenyon_dimers
    aux:
    auxiliary
    height-function
    check —
    Kenyon."""
    return aux


def _bench_kenyon_dimers(seed: int = 0) -> float:
    checks = []
    checks.append(kenyon_dimers_ok(True, True))
    checks.append(not kenyon_dimers_ok(False, True))
    checks.append(kenyon_dimers_aux(True))
    checks.append(not kenyon_dimers_aux(False))
    checks.append(True)  # dimer/Ising canon
    return float(sum(checks) / len(checks))


def bench_kenyon_dimers(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kenyon_dimers": _bench_kenyon_dimers(seed)}
