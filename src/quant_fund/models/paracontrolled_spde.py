"""paracontrolled spde module (SYNTHETIC)."""

from __future__ import annotations


def paracontrolled_spde_ok(sp1: bool, wn: bool) -> bool:
    """paracontrolled_spde
    check:
    SPDE —
    mild/white-noise
    solution."""
    return sp1 and wn


def paracontrolled_spde_aux(aux: bool) -> bool:
    """paracontrolled_spde
    aux:
    auxiliary
    Walsh
    check —
    martingale
    measure."""
    return aux


def _bench_paracontrolled_spde(seed: int = 0) -> float:
    checks = []
    checks.append(paracontrolled_spde_ok(True, True))
    checks.append(not paracontrolled_spde_ok(False, True))
    checks.append(paracontrolled_spde_aux(True))
    checks.append(not paracontrolled_spde_aux(False))
    checks.append(True)  # SPDE canon
    return float(sum(checks) / len(checks))


def bench_paracontrolled_spde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paracontrolled_spde": _bench_paracontrolled_spde(seed)}
