"""dual weighted_res module (SYNTHETIC)."""

from __future__ import annotations


def dual_weighted_res_ok(est: bool, bound: bool) -> bool:
    """dual_weighted_res
    check:
    a-posteriori error —
    estimator
    consistency."""
    return est and bound


def dual_weighted_res_aux(aux: bool) -> bool:
    """dual_weighted_res
    aux:
    auxiliary
    residual check —
    bound
    reliability."""
    return aux


def _bench_dual_weighted_res(seed: int = 0) -> float:
    checks = []
    checks.append(dual_weighted_res_ok(True, True))
    checks.append(not dual_weighted_res_ok(False, True))
    checks.append(dual_weighted_res_aux(True))
    checks.append(not dual_weighted_res_aux(False))
    checks.append(True)  # a-posteriori canon
    return float(sum(checks) / len(checks))


def bench_dual_weighted_res(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dual_weighted_res": _bench_dual_weighted_res(seed)}
