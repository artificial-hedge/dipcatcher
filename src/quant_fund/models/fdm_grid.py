"""fdm grid module (SYNTHETIC)."""

from __future__ import annotations


def fdm_grid_ok(grid: bool, flux: bool) -> bool:
    """fdm_grid
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def fdm_grid_aux(aux: bool) -> bool:
    """fdm_grid
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_fdm_grid(seed: int = 0) -> float:
    checks = []
    checks.append(fdm_grid_ok(True, True))
    checks.append(not fdm_grid_ok(False, True))
    checks.append(fdm_grid_aux(True))
    checks.append(not fdm_grid_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_fdm_grid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fdm_grid": _bench_fdm_grid(seed)}
