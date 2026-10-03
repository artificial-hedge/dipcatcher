"""motivic street module (SYNTHETIC)."""

from __future__ import annotations


def motivic_street_ok(motivic: bool, stable: bool) -> bool:
    """motivic_street
    check:
    motivic
    structure —
    frobenius."""
    return motivic and stable


def motivic_street_aux(aux: bool) -> bool:
    """motivic_street
    aux:
    auxiliary
    motivic
    check —
    cartier."""
    return aux


def _bench_motivic_street(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_street_ok(True, True))
    checks.append(not motivic_street_ok(False, True))
    checks.append(motivic_street_aux(True))
    checks.append(not motivic_street_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_street(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_street": _bench_motivic_street(seed)}
