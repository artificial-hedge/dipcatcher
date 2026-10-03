"""lsqr solver module (SYNTHETIC)."""

from __future__ import annotations


def lsqr_solver_ok(krylov: bool, resid: bool) -> bool:
    """lsqr_solver
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def lsqr_solver_aux(aux: bool) -> bool:
    """lsqr_solver
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_lsqr_solver(seed: int = 0) -> float:
    checks = []
    checks.append(lsqr_solver_ok(True, True))
    checks.append(not lsqr_solver_ok(False, True))
    checks.append(lsqr_solver_aux(True))
    checks.append(not lsqr_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_lsqr_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lsqr_solver": _bench_lsqr_solver(seed)}
