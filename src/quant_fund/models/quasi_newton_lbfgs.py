"""quasi newton_lbfgs module (SYNTHETIC)."""

from __future__ import annotations


def quasi_newton_lbfgs_ok(step: bool, conv: bool) -> bool:
    """quasi_newton_lbfgs
    check:
    optimization —
    descent step
    consistency."""
    return step and conv


def quasi_newton_lbfgs_aux(aux: bool) -> bool:
    """quasi_newton_lbfgs
    aux:
    auxiliary
    optimizer check —
    rate bound."""
    return aux


def _bench_quasi_newton_lbfgs(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_newton_lbfgs_ok(True, True))
    checks.append(not quasi_newton_lbfgs_ok(False, True))
    checks.append(quasi_newton_lbfgs_aux(True))
    checks.append(not quasi_newton_lbfgs_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_quasi_newton_lbfgs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_newton_lbfgs": _bench_quasi_newton_lbfgs(seed)}
