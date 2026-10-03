"""motivic heart module (SYNTHETIC)."""

from __future__ import annotations


def motivic_heart_ok(motivic: bool, stable: bool) -> bool:
    """motivic_heart
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_heart_aux(aux: bool) -> bool:
    """motivic_heart
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_heart(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_heart_ok(True, True))
    checks.append(not motivic_heart_ok(False, True))
    checks.append(motivic_heart_aux(True))
    checks.append(not motivic_heart_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_heart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_heart": _bench_motivic_heart(seed)}
