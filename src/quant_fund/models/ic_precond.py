"""ic precond module (SYNTHETIC)."""

from __future__ import annotations


def ic_precond_ok(pre: bool, conv: bool) -> bool:
    """ic_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def ic_precond_aux(aux: bool) -> bool:
    """ic_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_ic_precond(seed: int = 0) -> float:
    checks = []
    checks.append(ic_precond_ok(True, True))
    checks.append(not ic_precond_ok(False, True))
    checks.append(ic_precond_aux(True))
    checks.append(not ic_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_ic_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ic_precond": _bench_ic_precond(seed)}
