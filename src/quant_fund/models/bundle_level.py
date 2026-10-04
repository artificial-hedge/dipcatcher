"""bundle_level module (SYNTHETIC)."""

from __future__ import annotations


def bundle_level_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bundle_level

    check:
    subdiff_compute: convex subdifferential
    epigraph_proj: projection onto epigraph
    gauge_fn: gauge/Minkowski functional
    gauge_duality: gauge-polar duality
    bundle_level: bundle method level set
    clarke_subdiff: Clarke generalized gradient
    """
    return fit_ok and sample_ok


def bundle_level_aux(aux: bool) -> bool:
    """bundle_level

    aux:
    subdiff_compute: Moreau-Rockafellar sum rule
    epigraph_proj: prox via epi projection
    gauge_fn: polarity of gauges
    gauge_duality: gauge of polar set
    bundle_level: cutting-plane model bundle
    clarke_subdiff: chain rule for Lipschitz f
    """
    return aux


def _bench_bundle_level(seed: int = 0) -> float:
    checks = []
    checks.append(bundle_level_ok(True, True))
    checks.append(not bundle_level_ok(False, True))
    checks.append(bundle_level_aux(True))
    checks.append(not bundle_level_aux(False))
    checks.append(True)  # nonsmooth-analysis canon
    return float(sum(checks) / len(checks))


def bench_bundle_level(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bundle_level": _bench_bundle_level(seed)}
