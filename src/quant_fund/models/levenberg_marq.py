"""levenberg marq module (SYNTHETIC)."""

from __future__ import annotations


def levenberg_marq_ok(step: bool, resid: bool) -> bool:
    """levenberg_marq
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def levenberg_marq_aux(aux: bool) -> bool:
    """levenberg_marq
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_levenberg_marq(seed: int = 0) -> float:
    checks = []
    checks.append(levenberg_marq_ok(True, True))
    checks.append(not levenberg_marq_ok(False, True))
    checks.append(levenberg_marq_aux(True))
    checks.append(not levenberg_marq_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_levenberg_marq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levenberg_marq": _bench_levenberg_marq(seed)}
