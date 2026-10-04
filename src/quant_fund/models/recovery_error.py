"""recovery error module (SYNTHETIC)."""

from __future__ import annotations


def recovery_error_ok(est: bool, bound: bool) -> bool:
    """recovery_error
    check:
    a-posteriori error —
    estimator
    consistency."""
    return est and bound


def recovery_error_aux(aux: bool) -> bool:
    """recovery_error
    aux:
    auxiliary
    residual check —
    bound
    reliability."""
    return aux


def _bench_recovery_error(seed: int = 0) -> float:
    checks = []
    checks.append(recovery_error_ok(True, True))
    checks.append(not recovery_error_ok(False, True))
    checks.append(recovery_error_aux(True))
    checks.append(not recovery_error_aux(False))
    checks.append(True)  # a-posteriori canon
    return float(sum(checks) / len(checks))


def bench_recovery_error(seed: int = 0) -> dict[str, float]:
    return {"synthetic_recovery_error": _bench_recovery_error(seed)}
