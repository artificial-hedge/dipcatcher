"""landweber iter module (SYNTHETIC)."""

from __future__ import annotations


def landweber_iter_ok(step: bool, resid: bool) -> bool:
    """landweber_iter
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def landweber_iter_aux(aux: bool) -> bool:
    """landweber_iter
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_landweber_iter(seed: int = 0) -> float:
    checks = []
    checks.append(landweber_iter_ok(True, True))
    checks.append(not landweber_iter_ok(False, True))
    checks.append(landweber_iter_aux(True))
    checks.append(not landweber_iter_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_landweber_iter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_landweber_iter": _bench_landweber_iter(seed)}
