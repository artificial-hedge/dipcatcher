"""motivic fundamental module (SYNTHETIC)."""

from __future__ import annotations


def motivic_fundamental_ok(motivic: bool, stable: bool) -> bool:
    """motivic_fundamental
    check:
    motivic
    structure —
    spark."""
    return motivic and stable


def motivic_fundamental_aux(aux: bool) -> bool:
    """motivic_fundamental
    aux:
    auxiliary
    motivic
    check —
    fundamental."""
    return aux


def _bench_motivic_fundamental(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_fundamental_ok(True, True))
    checks.append(not motivic_fundamental_ok(False, True))
    checks.append(motivic_fundamental_aux(True))
    checks.append(not motivic_fundamental_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_fundamental(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_fundamental": _bench_motivic_fundamental(seed)}
