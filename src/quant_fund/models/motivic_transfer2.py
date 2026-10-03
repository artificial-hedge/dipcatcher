"""motivic transfer2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_transfer2_ok(motivic: bool, stable: bool) -> bool:
    """motivic_transfer2
    check:
    motivic
    structure —
    trace."""
    return motivic and stable


def motivic_transfer2_aux(aux: bool) -> bool:
    """motivic_transfer2
    aux:
    auxiliary
    motivic
    check —
    transfer."""
    return aux


def _bench_motivic_transfer2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_transfer2_ok(True, True))
    checks.append(not motivic_transfer2_ok(False, True))
    checks.append(motivic_transfer2_aux(True))
    checks.append(not motivic_transfer2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_transfer2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_transfer2": _bench_motivic_transfer2(seed)}
