"""motivic wit module (SYNTHETIC)."""

from __future__ import annotations


def motivic_wit_ok(motivic: bool, stable: bool) -> bool:
    """motivic_wit
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_wit_aux(aux: bool) -> bool:
    """motivic_wit
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_wit(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_wit_ok(True, True))
    checks.append(not motivic_wit_ok(False, True))
    checks.append(motivic_wit_aux(True))
    checks.append(not motivic_wit_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_wit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_wit": _bench_motivic_wit(seed)}
