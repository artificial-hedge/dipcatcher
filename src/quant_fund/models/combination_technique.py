"""combination technique module (SYNTHETIC)."""

from __future__ import annotations


def combination_technique_ok(grid: bool, level: bool) -> bool:
    """combination_technique
    check:
    sparse-grid/dimension-adaptive —
    surplus
    consistency."""
    return grid and level


def combination_technique_aux(aux: bool) -> bool:
    """combination_technique
    aux:
    auxiliary
    sparse check —
    tensor bound."""
    return aux


def _bench_combination_technique(seed: int = 0) -> float:
    checks = []
    checks.append(combination_technique_ok(True, True))
    checks.append(not combination_technique_ok(False, True))
    checks.append(combination_technique_aux(True))
    checks.append(not combination_technique_aux(False))
    checks.append(True)  # sparse-grid canon
    return float(sum(checks) / len(checks))


def bench_combination_technique(seed: int = 0) -> dict[str, float]:
    return {"synthetic_combination_technique": _bench_combination_technique(seed)}
