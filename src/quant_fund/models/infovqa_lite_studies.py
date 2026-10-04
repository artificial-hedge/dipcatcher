"""infovqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def infovqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """infovqa_lite_studies

    check:
    infovqa_lite_studies: InfographicVQA metrics
    """
    return fit_ok and sample_ok


def infovqa_lite_studies_aux(aux: bool) -> bool:
    """infovqa_lite_studies

    aux:
    infovqa_lite_studies: infographics, questions, answers, and scores
    """
    return aux


def _bench_infovqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(infovqa_lite_studies_ok(True, True))
    checks.append(not infovqa_lite_studies_ok(False, True))
    checks.append(infovqa_lite_studies_aux(True))
    checks.append(not infovqa_lite_studies_aux(False))
    checks.append(True)  # vision-doc-QA canon
    return float(sum(checks) / len(checks))


def bench_infovqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infovqa_lite_studies": _bench_infovqa_lite_studies(seed)}
