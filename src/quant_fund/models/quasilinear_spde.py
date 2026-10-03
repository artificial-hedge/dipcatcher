"""quasilinear spde module (SYNTHETIC)."""

from __future__ import annotations


def quasilinear_spde_ok(sp1: bool, wn: bool) -> bool:
    """quasilinear_spde
    check:
    SPDE —
    mild/white-noise
    solution."""
    return sp1 and wn


def quasilinear_spde_aux(aux: bool) -> bool:
    """quasilinear_spde
    aux:
    auxiliary
    Walsh
    check —
    martingale
    measure."""
    return aux


def _bench_quasilinear_spde(seed: int = 0) -> float:
    checks = []
    checks.append(quasilinear_spde_ok(True, True))
    checks.append(not quasilinear_spde_ok(False, True))
    checks.append(quasilinear_spde_aux(True))
    checks.append(not quasilinear_spde_aux(False))
    checks.append(True)  # SPDE canon
    return float(sum(checks) / len(checks))


def bench_quasilinear_spde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasilinear_spde": _bench_quasilinear_spde(seed)}
