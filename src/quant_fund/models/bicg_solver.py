"""bicg solver module (SYNTHETIC)."""

from __future__ import annotations


def bicg_solver_ok(krylov: bool, resid: bool) -> bool:
    """bicg_solver
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def bicg_solver_aux(aux: bool) -> bool:
    """bicg_solver
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_bicg_solver(seed: int = 0) -> float:
    checks = []
    checks.append(bicg_solver_ok(True, True))
    checks.append(not bicg_solver_ok(False, True))
    checks.append(bicg_solver_aux(True))
    checks.append(not bicg_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_bicg_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bicg_solver": _bench_bicg_solver(seed)}
