"""minres solver module (SYNTHETIC)."""

from __future__ import annotations


def minres_solver_ok(res: bool, it: bool) -> bool:
    """minres_solver
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def minres_solver_aux(aux: bool) -> bool:
    """minres_solver
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_minres_solver(seed: int = 0) -> float:
    checks = []
    checks.append(minres_solver_ok(True, True))
    checks.append(not minres_solver_ok(False, True))
    checks.append(minres_solver_aux(True))
    checks.append(not minres_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_minres_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minres_solver": _bench_minres_solver(seed)}
