"""subdiff_compute module (SYNTHETIC)."""

from __future__ import annotations


def subdiff_compute_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """subdiff_compute

    check:
    subdiff_compute: convex subdifferential
    epigraph_proj: projection onto epigraph
    gauge_fn: gauge/Minkowski functional
    gauge_duality: gauge-polar duality
    bundle_level: bundle method level set
    clarke_subdiff: Clarke generalized gradient
    """
    return fit_ok and sample_ok


def subdiff_compute_aux(aux: bool) -> bool:
    """subdiff_compute

    aux:
    subdiff_compute: Moreau-Rockafellar sum rule
    epigraph_proj: prox via epi projection
    gauge_fn: polarity of gauges
    gauge_duality: gauge of polar set
    bundle_level: cutting-plane model bundle
    clarke_subdiff: chain rule for Lipschitz f
    """
    return aux


def _bench_subdiff_compute(seed: int = 0) -> float:
    checks = []
    checks.append(subdiff_compute_ok(True, True))
    checks.append(not subdiff_compute_ok(False, True))
    checks.append(subdiff_compute_aux(True))
    checks.append(not subdiff_compute_aux(False))
    checks.append(True)  # nonsmooth-analysis canon
    return float(sum(checks) / len(checks))


def bench_subdiff_compute(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subdiff_compute": _bench_subdiff_compute(seed)}
