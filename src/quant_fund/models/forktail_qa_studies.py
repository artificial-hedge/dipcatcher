"""forktail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def forktail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forktail_qa_studies

    check:
    forktail_qa_studies: ForktailQA metrics
    """
    return fit_ok and sample_ok


def forktail_qa_studies_aux(aux: bool) -> bool:
    """forktail_qa_studies

    aux:
    forktail_qa_studies: forktails, wetlands, answers, and scores
    """
    return aux


def _bench_forktail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(forktail_qa_studies_ok(True, True))
    checks.append(not forktail_qa_studies_ok(False, True))
    checks.append(forktail_qa_studies_aux(True))
    checks.append(not forktail_qa_studies_aux(False))
    checks.append(True)  # dragonfly canon
    return float(sum(checks) / len(checks))


def bench_forktail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forktail_qa_studies": _bench_forktail_qa_studies(seed)}
