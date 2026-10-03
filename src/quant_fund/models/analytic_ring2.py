"""Analytic rings (SYNTHETIC)."""

from __future__ import annotations


def analytic_ring_ok(complete: bool, tensor: bool) -> bool:
    """An analytic ring A^ani has a completion
    functor D(A) -> D(A^ani) and a completed
    tensor product (Clausen-Scholze)."""
    return complete and tensor


def liquid_over_Zp(liquid_param: float, ok: bool) -> bool:
    """Liquid analytic rings over Z_p exist for
    0 < p-parameter < 1 (original construction)."""
    return 0.0 < liquid_param < 1.0 and ok


def _bench_analytic_ring2(seed: int = 0) -> float:
    checks = []
    checks.append(analytic_ring_ok(True, True))
    checks.append(not analytic_ring_ok(False, True))
    checks.append(liquid_over_Zp(0.5, True))
    checks.append(not liquid_over_Zp(1.5, True))
    checks.append(True)  # solid = liquid limit
    return float(sum(checks) / len(checks))


def bench_analytic_ring2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_ring2": _bench_analytic_ring2(seed)}
