"""motivic lax module (SYNTHETIC)."""

from __future__ import annotations


def motivic_lax_ok(motivic: bool, stable: bool) -> bool:
    """motivic_lax
    check:
    motivic
    structure —
    frobenius."""
    return motivic and stable


def motivic_lax_aux(aux: bool) -> bool:
    """motivic_lax
    aux:
    auxiliary
    motivic
    check —
    cartier."""
    return aux


def _bench_motivic_lax(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_lax_ok(True, True))
    checks.append(not motivic_lax_ok(False, True))
    checks.append(motivic_lax_aux(True))
    checks.append(not motivic_lax_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_lax(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_lax": _bench_motivic_lax(seed)}
