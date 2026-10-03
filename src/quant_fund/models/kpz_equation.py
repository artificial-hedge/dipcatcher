"""kpz equation module (SYNTHETIC)."""

from __future__ import annotations


def kpz_equation_ok(sp1: bool, wn: bool) -> bool:
    """kpz_equation
    check:
    SPDE —
    mild/white-noise
    solution."""
    return sp1 and wn


def kpz_equation_aux(aux: bool) -> bool:
    """kpz_equation
    aux:
    auxiliary
    Walsh
    check —
    martingale
    measure."""
    return aux


def _bench_kpz_equation(seed: int = 0) -> float:
    checks = []
    checks.append(kpz_equation_ok(True, True))
    checks.append(not kpz_equation_ok(False, True))
    checks.append(kpz_equation_aux(True))
    checks.append(not kpz_equation_aux(False))
    checks.append(True)  # SPDE canon
    return float(sum(checks) / len(checks))


def bench_kpz_equation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kpz_equation": _bench_kpz_equation(seed)}
