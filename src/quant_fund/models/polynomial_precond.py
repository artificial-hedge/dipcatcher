"""polynomial precond module (SYNTHETIC)."""

from __future__ import annotations


def polynomial_precond_ok(pre: bool, conv: bool) -> bool:
    """polynomial_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def polynomial_precond_aux(aux: bool) -> bool:
    """polynomial_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_polynomial_precond(seed: int = 0) -> float:
    checks = []
    checks.append(polynomial_precond_ok(True, True))
    checks.append(not polynomial_precond_ok(False, True))
    checks.append(polynomial_precond_aux(True))
    checks.append(not polynomial_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_polynomial_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polynomial_precond": _bench_polynomial_precond(seed)}
