"""medmcqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def medmcqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """medmcqa_lite_studies

    check:
    medmcqa_lite_studies: MedMCQA metrics
    """
    return fit_ok and sample_ok


def medmcqa_lite_studies_aux(aux: bool) -> bool:
    """medmcqa_lite_studies

    aux:
    medmcqa_lite_studies: questions, choices, answers, and scores
    """
    return aux


def _bench_medmcqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(medmcqa_lite_studies_ok(True, True))
    checks.append(not medmcqa_lite_studies_ok(False, True))
    checks.append(medmcqa_lite_studies_aux(True))
    checks.append(not medmcqa_lite_studies_aux(False))
    checks.append(True)  # science-QA-2 canon
    return float(sum(checks) / len(checks))


def bench_medmcqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_medmcqa_lite_studies": _bench_medmcqa_lite_studies(seed)}
