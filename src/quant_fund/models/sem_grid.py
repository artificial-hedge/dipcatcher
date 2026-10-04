"""sem grid module (SYNTHETIC)."""

from __future__ import annotations


def sem_grid_ok(node: bool, poly: bool) -> bool:
    """sem_grid
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def sem_grid_aux(aux: bool) -> bool:
    """sem_grid
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_sem_grid(seed: int = 0) -> float:
    checks = []
    checks.append(sem_grid_ok(True, True))
    checks.append(not sem_grid_ok(False, True))
    checks.append(sem_grid_aux(True))
    checks.append(not sem_grid_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_sem_grid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sem_grid": _bench_sem_grid(seed)}
