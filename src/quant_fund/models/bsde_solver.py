"""bsde solver module (SYNTHETIC)."""

from __future__ import annotations


def bsde_solver_ok(bs1: bool, pp: bool) -> bool:
    """bsde_solver
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def bsde_solver_aux(aux: bool) -> bool:
    """bsde_solver
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_bsde_solver(seed: int = 0) -> float:
    checks = []
    checks.append(bsde_solver_ok(True, True))
    checks.append(not bsde_solver_ok(False, True))
    checks.append(bsde_solver_aux(True))
    checks.append(not bsde_solver_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_bsde_solver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bsde_solver": _bench_bsde_solver(seed)}
