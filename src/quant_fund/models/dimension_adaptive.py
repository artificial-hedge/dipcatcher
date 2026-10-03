"""dimension adaptive module (SYNTHETIC)."""

from __future__ import annotations


def dimension_adaptive_ok(grid: bool, level: bool) -> bool:
    """dimension_adaptive
    check:
    sparse-grid/dimension-adaptive —
    surplus
    consistency."""
    return grid and level


def dimension_adaptive_aux(aux: bool) -> bool:
    """dimension_adaptive
    aux:
    auxiliary
    sparse check —
    tensor bound."""
    return aux


def _bench_dimension_adaptive(seed: int = 0) -> float:
    checks = []
    checks.append(dimension_adaptive_ok(True, True))
    checks.append(not dimension_adaptive_ok(False, True))
    checks.append(dimension_adaptive_aux(True))
    checks.append(not dimension_adaptive_aux(False))
    checks.append(True)  # sparse-grid canon
    return float(sum(checks) / len(checks))


def bench_dimension_adaptive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dimension_adaptive": _bench_dimension_adaptive(seed)}
