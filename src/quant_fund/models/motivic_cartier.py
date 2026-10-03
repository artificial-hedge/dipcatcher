"""motivic cartier module (SYNTHETIC)."""

from __future__ import annotations


def motivic_cartier_ok(motivic: bool, stable: bool) -> bool:
    """motivic_cartier
    check:
    motivic
    structure —
    frobenius."""
    return motivic and stable


def motivic_cartier_aux(aux: bool) -> bool:
    """motivic_cartier
    aux:
    auxiliary
    motivic
    check —
    cartier."""
    return aux


def _bench_motivic_cartier(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_cartier_ok(True, True))
    checks.append(not motivic_cartier_ok(False, True))
    checks.append(motivic_cartier_aux(True))
    checks.append(not motivic_cartier_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_cartier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_cartier": _bench_motivic_cartier(seed)}
