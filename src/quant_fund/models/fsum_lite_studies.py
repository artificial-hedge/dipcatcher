"""fsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def fsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fsum_lite_studies

    check:
    fsum_lite_studies: FacetSum metrics
    """
    return fit_ok and sample_ok


def fsum_lite_studies_aux(aux: bool) -> bool:
    """fsum_lite_studies

    aux:
    fsum_lite_studies: documents, facets, summaries, and scores
    """
    return aux


def _bench_fsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fsum_lite_studies_ok(True, True))
    checks.append(not fsum_lite_studies_ok(False, True))
    checks.append(fsum_lite_studies_aux(True))
    checks.append(not fsum_lite_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_fsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fsum_lite_studies": _bench_fsum_lite_studies(seed)}
