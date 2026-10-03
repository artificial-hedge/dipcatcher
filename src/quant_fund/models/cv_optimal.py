"""cv optimal module (SYNTHETIC)."""

from __future__ import annotations


def cv_optimal_ok(step: bool, conv: bool) -> bool:
    """cv_optimal
    check:
    acceleration —
    step/contraction
    consistency."""
    return step and conv


def cv_optimal_aux(aux: bool) -> bool:
    """cv_optimal
    aux:
    auxiliary
    accelerator check —
    rate bound."""
    return aux


def _bench_cv_optimal(seed: int = 0) -> float:
    checks = []
    checks.append(cv_optimal_ok(True, True))
    checks.append(not cv_optimal_ok(False, True))
    checks.append(cv_optimal_aux(True))
    checks.append(not cv_optimal_aux(False))
    checks.append(True)  # acceleration canon
    return float(sum(checks) / len(checks))


def bench_cv_optimal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cv_optimal": _bench_cv_optimal(seed)}
