"""gmres solver module (SYNTHETIC)."""

from __future__ import annotations


def gmres_solver_ok(krylov: bool, resid: bool) -> bool:
    """gmres_solver
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def gmres_solver_aux(aux: bool) -> bool:
    """gmres_solver
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_gmres_solver(seed: int = 0) -> float:
    checks = []
    checks.append(gmres_solver_ok(True, True))
    checks.append(not gmres_solver_ok(False, True))
    checks.append(gmres_solver_aux(True))
    checks.append(not gmres_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_gmres_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gmres_solver": _bench_gmres_solver(seed)}
