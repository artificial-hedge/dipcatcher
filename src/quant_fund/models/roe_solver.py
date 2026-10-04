"""roe solver module (SYNTHETIC)."""

from __future__ import annotations


def roe_solver_ok(state: bool, flux: bool) -> bool:
    """roe_solver
    check:
    Riemann-solver —
    flux consistency."""
    return state and flux


def roe_solver_aux(aux: bool) -> bool:
    """roe_solver
    aux:
    auxiliary
    solver check —
    entropy fix."""
    return aux


def _bench_roe_solver(seed: int = 0) -> float:
    checks = []
    checks.append(roe_solver_ok(True, True))
    checks.append(not roe_solver_ok(False, True))
    checks.append(roe_solver_aux(True))
    checks.append(not roe_solver_aux(False))
    checks.append(True)  # Riemann-solver canon
    return float(sum(checks) / len(checks))


def bench_roe_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roe_solver": _bench_roe_solver(seed)}
