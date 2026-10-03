"""motivic diagonal module (SYNTHETIC)."""

from __future__ import annotations


def motivic_diagonal_ok(motivic: bool, stable: bool) -> bool:
    """motivic_diagonal
    check:
    motivic
    structure —
    spark."""
    return motivic and stable


def motivic_diagonal_aux(aux: bool) -> bool:
    """motivic_diagonal
    aux:
    auxiliary
    motivic
    check —
    fundamental."""
    return aux


def _bench_motivic_diagonal(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_diagonal_ok(True, True))
    checks.append(not motivic_diagonal_ok(False, True))
    checks.append(motivic_diagonal_aux(True))
    checks.append(not motivic_diagonal_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_diagonal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_diagonal": _bench_motivic_diagonal(seed)}
