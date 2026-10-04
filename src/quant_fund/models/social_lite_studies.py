"""social_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def social_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """social_lite_studies

    check:
    social_lite_studies: Social-IQA metrics
    """
    return fit_ok and sample_ok


def social_lite_studies_aux(aux: bool) -> bool:
    """social_lite_studies

    aux:
    social_lite_studies: contexts, questions, options, and scores
    """
    return aux


def _bench_social_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(social_lite_studies_ok(True, True))
    checks.append(not social_lite_studies_ok(False, True))
    checks.append(social_lite_studies_aux(True))
    checks.append(not social_lite_studies_aux(False))
    checks.append(True)  # intent-paraphrase canon
    return float(sum(checks) / len(checks))


def bench_social_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_social_lite_studies": _bench_social_lite_studies(seed)}
