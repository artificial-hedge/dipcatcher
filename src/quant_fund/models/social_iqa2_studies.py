"""social_iqa2_studies module (SYNTHETIC)."""

from __future__ import annotations


def social_iqa2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """social_iqa2_studies

    check:
    social_iqa2_studies: Social-IQA commonsense metrics
    """
    return fit_ok and sample_ok


def social_iqa2_studies_aux(aux: bool) -> bool:
    """social_iqa2_studies

    aux:
    social_iqa2_studies: contexts, questions, options, and accuracies
    """
    return aux


def _bench_social_iqa2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(social_iqa2_studies_ok(True, True))
    checks.append(not social_iqa2_studies_ok(False, True))
    checks.append(social_iqa2_studies_aux(True))
    checks.append(not social_iqa2_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_social_iqa2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_social_iqa2_studies": _bench_social_iqa2_studies(seed)}
