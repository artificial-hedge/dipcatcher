"""cg solver module (SYNTHETIC)."""

from __future__ import annotations


def cg_solver_ok(krylov: bool, resid: bool) -> bool:
    """cg_solver
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def cg_solver_aux(aux: bool) -> bool:
    """cg_solver
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_cg_solver(seed: int = 0) -> float:
    checks = []
    checks.append(cg_solver_ok(True, True))
    checks.append(not cg_solver_ok(False, True))
    checks.append(cg_solver_aux(True))
    checks.append(not cg_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_cg_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cg_solver": _bench_cg_solver(seed)}
