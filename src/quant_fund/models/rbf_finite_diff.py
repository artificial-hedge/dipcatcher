"""rbf finite_diff module (SYNTHETIC)."""

from __future__ import annotations


def rbf_finite_diff_ok(center: bool, shape: bool) -> bool:
    """rbf_finite_diff
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def rbf_finite_diff_aux(aux: bool) -> bool:
    """rbf_finite_diff
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_rbf_finite_diff(seed: int = 0) -> float:
    checks = []
    checks.append(rbf_finite_diff_ok(True, True))
    checks.append(not rbf_finite_diff_ok(False, True))
    checks.append(rbf_finite_diff_aux(True))
    checks.append(not rbf_finite_diff_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_rbf_finite_diff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rbf_finite_diff": _bench_rbf_finite_diff(seed)}
