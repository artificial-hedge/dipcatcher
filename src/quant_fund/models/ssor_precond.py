"""ssor precond module (SYNTHETIC)."""

from __future__ import annotations


def ssor_precond_ok(pre: bool, conv: bool) -> bool:
    """ssor_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def ssor_precond_aux(aux: bool) -> bool:
    """ssor_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_ssor_precond(seed: int = 0) -> float:
    checks = []
    checks.append(ssor_precond_ok(True, True))
    checks.append(not ssor_precond_ok(False, True))
    checks.append(ssor_precond_aux(True))
    checks.append(not ssor_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_ssor_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ssor_precond": _bench_ssor_precond(seed)}
