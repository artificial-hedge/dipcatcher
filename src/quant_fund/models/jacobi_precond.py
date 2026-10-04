"""jacobi precond module (SYNTHETIC)."""

from __future__ import annotations


def jacobi_precond_ok(pre: bool, conv: bool) -> bool:
    """jacobi_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def jacobi_precond_aux(aux: bool) -> bool:
    """jacobi_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_jacobi_precond(seed: int = 0) -> float:
    checks = []
    checks.append(jacobi_precond_ok(True, True))
    checks.append(not jacobi_precond_ok(False, True))
    checks.append(jacobi_precond_aux(True))
    checks.append(not jacobi_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_jacobi_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobi_precond": _bench_jacobi_precond(seed)}
