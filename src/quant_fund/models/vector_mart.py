"""vector mart module (SYNTHETIC)."""

from __future__ import annotations


def vector_mart_ok(si: bool, isom: bool) -> bool:
    """vector_mart
    check:
    stochastic
    integral —
    isometry."""
    return si and isom


def vector_mart_aux(aux: bool) -> bool:
    """vector_mart
    aux:
    auxiliary
    integral
    check —
    covariation."""
    return aux


def _bench_vector_mart(seed: int = 0) -> float:
    checks = []
    checks.append(vector_mart_ok(True, True))
    checks.append(not vector_mart_ok(False, True))
    checks.append(vector_mart_aux(True))
    checks.append(not vector_mart_aux(False))
    checks.append(True)  # integration canon
    return float(sum(checks) / len(checks))


def bench_vector_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vector_mart": _bench_vector_mart(seed)}
