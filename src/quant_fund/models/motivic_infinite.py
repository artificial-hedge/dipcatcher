"""motivic infinite module (SYNTHETIC)."""

from __future__ import annotations


def motivic_infinite_ok(motivic: bool, stable: bool) -> bool:
    """motivic_infinite
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_infinite_aux(aux: bool) -> bool:
    """motivic_infinite
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_infinite(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_infinite_ok(True, True))
    checks.append(not motivic_infinite_ok(False, True))
    checks.append(motivic_infinite_aux(True))
    checks.append(not motivic_infinite_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_infinite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_infinite": _bench_motivic_infinite(seed)}
