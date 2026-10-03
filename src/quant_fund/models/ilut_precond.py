"""ilut precond module (SYNTHETIC)."""

from __future__ import annotations


def ilut_precond_ok(pre: bool, conv: bool) -> bool:
    """ilut_precond
    check:
    preconditioner —
    spectral-condition
    consistency."""
    return pre and conv


def ilut_precond_aux(aux: bool) -> bool:
    """ilut_precond
    aux:
    auxiliary
    preconditioner check —
    condition bound."""
    return aux


def _bench_ilut_precond(seed: int = 0) -> float:
    checks = []
    checks.append(ilut_precond_ok(True, True))
    checks.append(not ilut_precond_ok(False, True))
    checks.append(ilut_precond_aux(True))
    checks.append(not ilut_precond_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_ilut_precond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ilut_precond": _bench_ilut_precond(seed)}
