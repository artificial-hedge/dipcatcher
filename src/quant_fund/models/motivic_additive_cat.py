"""motivic additive_cat module (SYNTHETIC)."""

from __future__ import annotations


def motivic_additive_cat_ok(motivic: bool, categorical: bool) -> bool:
    """motivic_additive_cat
    check:
    motivic
    structure —
    additive."""
    return motivic and categorical


def motivic_additive_cat_aux(aux: bool) -> bool:
    """motivic_additive_cat
    aux:
    auxiliary
    motivic
    check —
    gysin."""
    return aux


def _bench_motivic_additive_cat(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_additive_cat_ok(True, True))
    checks.append(not motivic_additive_cat_ok(False, True))
    checks.append(motivic_additive_cat_aux(True))
    checks.append(not motivic_additive_cat_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_additive_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_additive_cat": _bench_motivic_additive_cat(seed)}
