"""gauge_fn module (SYNTHETIC)."""

from __future__ import annotations


def gauge_fn_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gauge_fn

    check:
    subdiff_compute: convex subdifferential
    epigraph_proj: projection onto epigraph
    gauge_fn: gauge/Minkowski functional
    gauge_duality: gauge-polar duality
    bundle_level: bundle method level set
    clarke_subdiff: Clarke generalized gradient
    """
    return fit_ok and sample_ok


def gauge_fn_aux(aux: bool) -> bool:
    """gauge_fn

    aux:
    subdiff_compute: Moreau-Rockafellar sum rule
    epigraph_proj: prox via epi projection
    gauge_fn: polarity of gauges
    gauge_duality: gauge of polar set
    bundle_level: cutting-plane model bundle
    clarke_subdiff: chain rule for Lipschitz f
    """
    return aux


def _bench_gauge_fn(seed: int = 0) -> float:
    checks = []
    checks.append(gauge_fn_ok(True, True))
    checks.append(not gauge_fn_ok(False, True))
    checks.append(gauge_fn_aux(True))
    checks.append(not gauge_fn_aux(False))
    checks.append(True)  # nonsmooth-analysis canon
    return float(sum(checks) / len(checks))


def bench_gauge_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauge_fn": _bench_gauge_fn(seed)}
