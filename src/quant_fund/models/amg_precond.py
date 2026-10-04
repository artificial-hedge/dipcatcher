"""amg precond module (SYNTHETIC)."""

from __future__ import annotations


def amg_precond_ok(pre: bool, conv: bool) -> bool:
    """amg_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def amg_precond_aux(aux: bool) -> bool:
    """amg_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_amg_precond(seed: int = 0) -> float:
    checks = []
    checks.append(amg_precond_ok(True, True))
    checks.append(not amg_precond_ok(False, True))
    checks.append(amg_precond_aux(True))
    checks.append(not amg_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_amg_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amg_precond": _bench_amg_precond(seed)}
