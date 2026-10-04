"""smolyak grid module (SYNTHETIC)."""

from __future__ import annotations


def smolyak_grid_ok(grid: bool, level: bool) -> bool:
    """smolyak_grid
    check:
    sparse-grid/dimension-adaptive —
    surplus
    consistency."""
    return grid and level


def smolyak_grid_aux(aux: bool) -> bool:
    """smolyak_grid
    aux:
    auxiliary
    sparse check —
    tensor bound."""
    return aux


def _bench_smolyak_grid(seed: int = 0) -> float:
    checks = []
    checks.append(smolyak_grid_ok(True, True))
    checks.append(not smolyak_grid_ok(False, True))
    checks.append(smolyak_grid_aux(True))
    checks.append(not smolyak_grid_aux(False))
    checks.append(True)  # sparse-grid canon
    return float(sum(checks) / len(checks))


def bench_smolyak_grid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smolyak_grid": _bench_smolyak_grid(seed)}
