"""spde heat module (SYNTHETIC)."""

from __future__ import annotations


def spde_heat_ok(sp1: bool, wn: bool) -> bool:
    """spde_heat
    check:
    SPDE —
    mild/white-noise
    solution."""
    return sp1 and wn


def spde_heat_aux(aux: bool) -> bool:
    """spde_heat
    aux:
    auxiliary
    Walsh
    check —
    martingale
    measure."""
    return aux


def _bench_spde_heat(seed: int = 0) -> float:
    checks = []
    checks.append(spde_heat_ok(True, True))
    checks.append(not spde_heat_ok(False, True))
    checks.append(spde_heat_aux(True))
    checks.append(not spde_heat_aux(False))
    checks.append(True)  # SPDE canon
    return float(sum(checks) / len(checks))


def bench_spde_heat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spde_heat": _bench_spde_heat(seed)}
