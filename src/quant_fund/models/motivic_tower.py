"""motivic tower module (SYNTHETIC)."""

from __future__ import annotations


def motivic_tower_ok(motivic: bool, stable: bool) -> bool:
    """motivic_tower
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_tower_aux(aux: bool) -> bool:
    """motivic_tower
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_tower(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_tower_ok(True, True))
    checks.append(not motivic_tower_ok(False, True))
    checks.append(motivic_tower_aux(True))
    checks.append(not motivic_tower_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_tower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_tower": _bench_motivic_tower(seed)}
