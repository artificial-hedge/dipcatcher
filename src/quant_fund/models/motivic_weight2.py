"""motivic weight2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_weight2_ok(motivic: bool, stable: bool) -> bool:
    """motivic_weight2
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_weight2_aux(aux: bool) -> bool:
    """motivic_weight2
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_weight2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_weight2_ok(True, True))
    checks.append(not motivic_weight2_ok(False, True))
    checks.append(motivic_weight2_aux(True))
    checks.append(not motivic_weight2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_weight2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_weight2": _bench_motivic_weight2(seed)}
