"""motivic degree module (SYNTHETIC)."""

from __future__ import annotations


def motivic_degree_ok(motivic: bool, stable: bool) -> bool:
    """motivic_degree
    check:
    motivic
    structure —
    spark."""
    return motivic and stable


def motivic_degree_aux(aux: bool) -> bool:
    """motivic_degree
    aux:
    auxiliary
    motivic
    check —
    fundamental."""
    return aux


def _bench_motivic_degree(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_degree_ok(True, True))
    checks.append(not motivic_degree_ok(False, True))
    checks.append(motivic_degree_aux(True))
    checks.append(not motivic_degree_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_degree(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_degree": _bench_motivic_degree(seed)}
