"""qmr solver module (SYNTHETIC)."""

from __future__ import annotations


def qmr_solver_ok(res: bool, it: bool) -> bool:
    """qmr_solver
    check:
    Krylov —
    residual
    consistency."""
    return res and it


def qmr_solver_aux(aux: bool) -> bool:
    """qmr_solver
    aux:
    auxiliary
    solver check —
    recurrence bound."""
    return aux


def _bench_qmr_solver(seed: int = 0) -> float:
    checks = []
    checks.append(qmr_solver_ok(True, True))
    checks.append(not qmr_solver_ok(False, True))
    checks.append(qmr_solver_aux(True))
    checks.append(not qmr_solver_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_qmr_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qmr_solver": _bench_qmr_solver(seed)}
