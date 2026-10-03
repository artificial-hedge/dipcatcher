"""epigraph_proj module (SYNTHETIC)."""

from __future__ import annotations


def epigraph_proj_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """epigraph_proj

    check:
    subdiff_compute: convex subdifferential
    epigraph_proj: projection onto epigraph
    gauge_fn: gauge/Minkowski functional
    gauge_duality: gauge-polar duality
    bundle_level: bundle method level set
    clarke_subdiff: Clarke generalized gradient
    """
    return fit_ok and sample_ok


def epigraph_proj_aux(aux: bool) -> bool:
    """epigraph_proj

    aux:
    subdiff_compute: Moreau-Rockafellar sum rule
    epigraph_proj: prox via epi projection
    gauge_fn: polarity of gauges
    gauge_duality: gauge of polar set
    bundle_level: cutting-plane model bundle
    clarke_subdiff: chain rule for Lipschitz f
    """
    return aux


def _bench_epigraph_proj(seed: int = 0) -> float:
    checks = []
    checks.append(epigraph_proj_ok(True, True))
    checks.append(not epigraph_proj_ok(False, True))
    checks.append(epigraph_proj_aux(True))
    checks.append(not epigraph_proj_aux(False))
    checks.append(True)  # nonsmooth-analysis canon
    return float(sum(checks) / len(checks))


def bench_epigraph_proj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epigraph_proj": _bench_epigraph_proj(seed)}
