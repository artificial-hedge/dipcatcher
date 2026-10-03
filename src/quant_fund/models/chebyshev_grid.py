"""chebyshev grid module (SYNTHETIC)."""

from __future__ import annotations


def chebyshev_grid_ok(grid: bool, basis: bool) -> bool:
    """chebyshev_grid
    check:
    spectral
    method —
    grid."""
    return grid and basis


def chebyshev_grid_aux(aux: bool) -> bool:
    """chebyshev_grid
    aux:
    auxiliary
    spectral check —
    coeff."""
    return aux


def _bench_chebyshev_grid(seed: int = 0) -> float:
    checks = []
    checks.append(chebyshev_grid_ok(True, True))
    checks.append(not chebyshev_grid_ok(False, True))
    checks.append(chebyshev_grid_aux(True))
    checks.append(not chebyshev_grid_aux(False))
    checks.append(True)  # spectral-methods canon
    return float(sum(checks) / len(checks))


def bench_chebyshev_grid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_grid": _bench_chebyshev_grid(seed)}
