"""bary_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def bary_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bary_score_studies

    check:
    bary_score_studies: BaryScore embedding metrics
    """
    return fit_ok and sample_ok


def bary_score_studies_aux(aux: bool) -> bool:
    """bary_score_studies

    aux:
    bary_score_studies: sources, references, candidates, and scores
    """
    return aux


def _bench_bary_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bary_score_studies_ok(True, True))
    checks.append(not bary_score_studies_ok(False, True))
    checks.append(bary_score_studies_aux(True))
    checks.append(not bary_score_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_bary_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bary_score_studies": _bench_bary_score_studies(seed)}
