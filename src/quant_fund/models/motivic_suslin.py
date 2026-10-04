"""motivic suslin module (SYNTHETIC)."""

from __future__ import annotations


def motivic_suslin_ok(motivic: bool, stable: bool) -> bool:
    """motivic_suslin
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_suslin_aux(aux: bool) -> bool:
    """motivic_suslin
    aux:
    auxiliary
    motivic
    check —
    slice."""
    return aux


def _bench_motivic_suslin(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_suslin_ok(True, True))
    checks.append(not motivic_suslin_ok(False, True))
    checks.append(motivic_suslin_aux(True))
    checks.append(not motivic_suslin_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_suslin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_suslin": _bench_motivic_suslin(seed)}
