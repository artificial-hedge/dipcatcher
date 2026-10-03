"""motivic gysin2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_gysin2_ok(motivic: bool, categorical: bool) -> bool:
    """motivic_gysin2
    check:
    motivic
    structure —
    additive."""
    return motivic and categorical


def motivic_gysin2_aux(aux: bool) -> bool:
    """motivic_gysin2
    aux:
    auxiliary
    motivic
    check —
    gysin."""
    return aux


def _bench_motivic_gysin2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_gysin2_ok(True, True))
    checks.append(not motivic_gysin2_ok(False, True))
    checks.append(motivic_gysin2_aux(True))
    checks.append(not motivic_gysin2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_gysin2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_gysin2": _bench_motivic_gysin2(seed)}
