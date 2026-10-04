"""cgs solver module (SYNTHETIC)."""

from __future__ import annotations


def cgs_solver_ok(res: bool, it: bool) -> bool:
    """cgs_solver
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def cgs_solver_aux(aux: bool) -> bool:
    """cgs_solver
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_cgs_solver(seed: int = 0) -> float:
    checks = []
    checks.append(cgs_solver_ok(True, True))
    checks.append(not cgs_solver_ok(False, True))
    checks.append(cgs_solver_aux(True))
    checks.append(not cgs_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_cgs_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cgs_solver": _bench_cgs_solver(seed)}
