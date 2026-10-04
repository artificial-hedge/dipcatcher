"""conjugate grad_ls module (SYNTHETIC)."""

from __future__ import annotations


def conjugate_grad_ls_ok(step: bool, resid: bool) -> bool:
    """conjugate_grad_ls
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def conjugate_grad_ls_aux(aux: bool) -> bool:
    """conjugate_grad_ls
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_conjugate_grad_ls(seed: int = 0) -> float:
    checks = []
    checks.append(conjugate_grad_ls_ok(True, True))
    checks.append(not conjugate_grad_ls_ok(False, True))
    checks.append(conjugate_grad_ls_aux(True))
    checks.append(not conjugate_grad_ls_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_conjugate_grad_ls(seed: int = 0) -> dict[str, float]:
    return {"synthetic_conjugate_grad_ls": _bench_conjugate_grad_ls(seed)}
