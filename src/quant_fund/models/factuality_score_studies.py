"""factuality_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def factuality_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """factuality_score_studies

    check:
    factuality_score_studies: sentence-level factual consistency/scores and errors
    """
    return fit_ok and sample_ok


def factuality_score_studies_aux(aux: bool) -> bool:
    """factuality_score_studies

    aux:
    factuality_score_studies: QAG/QAFactEval-style QA verification/questions and answers
    """
    return aux


def _bench_factuality_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(factuality_score_studies_ok(True, True))
    checks.append(not factuality_score_studies_ok(False, True))
    checks.append(factuality_score_studies_aux(True))
    checks.append(not factuality_score_studies_aux(False))
    checks.append(True)  # grounding/hallucination canon
    return float(sum(checks) / len(checks))


def bench_factuality_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_factuality_score_studies": _bench_factuality_score_studies(seed)}
