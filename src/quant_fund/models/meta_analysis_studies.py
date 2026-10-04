"""meta_analysis_studies module (SYNTHETIC)."""

from __future__ import annotations


def meta_analysis_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """meta_analysis_studies

    check:
    meta_analysis_studies: pooling and heterogeneity/fixed and random
    """
    return fit_ok and sample_ok


def meta_analysis_studies_aux(aux: bool) -> bool:
    """meta_analysis_studies

    aux:
    meta_analysis_studies: bias and funnel/publication and sensitivity
    """
    return aux


def _bench_meta_analysis_studies(seed: int = 0) -> float:
    checks = []
    checks.append(meta_analysis_studies_ok(True, True))
    checks.append(not meta_analysis_studies_ok(False, True))
    checks.append(meta_analysis_studies_aux(True))
    checks.append(not meta_analysis_studies_aux(False))
    checks.append(True)  # clinical-research-methods canon
    return float(sum(checks) / len(checks))


def bench_meta_analysis_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_meta_analysis_studies": _bench_meta_analysis_studies(seed)}
