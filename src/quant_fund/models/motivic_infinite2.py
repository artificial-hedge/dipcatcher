"""motivic infinite2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_infinite2_ok(motivic: bool, stable: bool) -> bool:
    """motivic_infinite2
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_infinite2_aux(aux: bool) -> bool:
    """motivic_infinite2
    aux:
    auxiliary
    motivic
    check —
    residue."""
    return aux


def _bench_motivic_infinite2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_infinite2_ok(True, True))
    checks.append(not motivic_infinite2_ok(False, True))
    checks.append(motivic_infinite2_aux(True))
    checks.append(not motivic_infinite2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_infinite2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_infinite2": _bench_motivic_infinite2(seed)}
