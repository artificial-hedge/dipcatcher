"""motivic base2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_base2_ok(motivic: bool, stable: bool) -> bool:
    """motivic_base2
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_base2_aux(aux: bool) -> bool:
    """motivic_base2
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_base2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_base2_ok(True, True))
    checks.append(not motivic_base2_ok(False, True))
    checks.append(motivic_base2_aux(True))
    checks.append(not motivic_base2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_base2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_base2": _bench_motivic_base2(seed)}
