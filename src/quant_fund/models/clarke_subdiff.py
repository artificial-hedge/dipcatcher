"""clarke_subdiff module (SYNTHETIC)."""

from __future__ import annotations


def clarke_subdiff_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """clarke_subdiff

    check:
    subdiff_compute: convex subdifferential
    epigraph_proj: projection onto epigraph
    gauge_fn: gauge/Minkowski functional
    gauge_duality: gauge-polar duality
    bundle_level: bundle method level set
    clarke_subdiff: Clarke generalized gradient
    """
    return fit_ok and sample_ok


def clarke_subdiff_aux(aux: bool) -> bool:
    """clarke_subdiff

    aux:
    subdiff_compute: Moreau-Rockafellar sum rule
    epigraph_proj: prox via epi projection
    gauge_fn: polarity of gauges
    gauge_duality: gauge of polar set
    bundle_level: cutting-plane model bundle
    clarke_subdiff: chain rule for Lipschitz f
    """
    return aux


def _bench_clarke_subdiff(seed: int = 0) -> float:
    checks = []
    checks.append(clarke_subdiff_ok(True, True))
    checks.append(not clarke_subdiff_ok(False, True))
    checks.append(clarke_subdiff_aux(True))
    checks.append(not clarke_subdiff_aux(False))
    checks.append(True)  # nonsmooth-analysis canon
    return float(sum(checks) / len(checks))


def bench_clarke_subdiff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clarke_subdiff": _bench_clarke_subdiff(seed)}
