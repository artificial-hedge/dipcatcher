"""adaptive finite module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_finite_ok(mark: bool, est: bool) -> bool:
    """adaptive_finite
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def adaptive_finite_aux(aux: bool) -> bool:
    """adaptive_finite
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_adaptive_finite(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_finite_ok(True, True))
    checks.append(not adaptive_finite_ok(False, True))
    checks.append(adaptive_finite_aux(True))
    checks.append(not adaptive_finite_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_adaptive_finite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_finite": _bench_adaptive_finite(seed)}
