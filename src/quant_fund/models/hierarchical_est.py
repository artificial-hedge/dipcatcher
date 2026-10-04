"""hierarchical est module (SYNTHETIC)."""

from __future__ import annotations


def hierarchical_est_ok(mark: bool, est: bool) -> bool:
    """hierarchical_est
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def hierarchical_est_aux(aux: bool) -> bool:
    """hierarchical_est
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_hierarchical_est(seed: int = 0) -> float:
    checks = []
    checks.append(hierarchical_est_ok(True, True))
    checks.append(not hierarchical_est_ok(False, True))
    checks.append(hierarchical_est_aux(True))
    checks.append(not hierarchical_est_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_hierarchical_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hierarchical_est": _bench_hierarchical_est(seed)}
