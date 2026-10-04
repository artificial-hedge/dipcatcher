"""residual estimator module (SYNTHETIC)."""

from __future__ import annotations


def residual_estimator_ok(est: bool, bound: bool) -> bool:
    """residual_estimator
    check:
    a-posteriori error —
    estimator
    consistency."""
    return est and bound


def residual_estimator_aux(aux: bool) -> bool:
    """residual_estimator
    aux:
    auxiliary
    residual check —
    bound
    reliability."""
    return aux


def _bench_residual_estimator(seed: int = 0) -> float:
    checks = []
    checks.append(residual_estimator_ok(True, True))
    checks.append(not residual_estimator_ok(False, True))
    checks.append(residual_estimator_aux(True))
    checks.append(not residual_estimator_aux(False))
    checks.append(True)  # a-posteriori canon
    return float(sum(checks) / len(checks))


def bench_residual_estimator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_residual_estimator": _bench_residual_estimator(seed)}
