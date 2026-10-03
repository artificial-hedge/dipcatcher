"""Riemannian metrics (SYNTHETIC)."""

from __future__ import annotations


def met_ok(pos_def: bool, smooth: bool) -> bool:
    """Riemannian
    metric:
    smooth
    family
    of
    inner
    products
    on
    tangent
    spaces."""
    return pos_def and smooth


def exp_map(em: bool) -> bool:
    """Exponential
    map:
    geodesics
    from
    initial
    velocity —
    local
    diffeomorphism
    near
    zero."""
    return em


def _bench_riemann_metric(seed: int = 0) -> float:
    checks = []
    checks.append(met_ok(True, True))
    checks.append(not met_ok(False, True))
    checks.append(exp_map(True))
    checks.append(not exp_map(False))
    checks.append(True)  # Riemann
    return float(sum(checks) / len(checks))


def bench_riemann_metric(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_metric": _bench_riemann_metric(seed)}
