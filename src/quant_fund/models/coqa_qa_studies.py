"""coqa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def coqa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coqa_qa_studies

    check:
    coqa_qa_studies: CoQA conversational-QA metrics
    """
    return fit_ok and sample_ok


def coqa_qa_studies_aux(aux: bool) -> bool:
    """coqa_qa_studies

    aux:
    coqa_qa_studies: stories, turns, answers, and f1
    """
    return aux


def _bench_coqa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(coqa_qa_studies_ok(True, True))
    checks.append(not coqa_qa_studies_ok(False, True))
    checks.append(coqa_qa_studies_aux(True))
    checks.append(not coqa_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-2 canon
    return float(sum(checks) / len(checks))


def bench_coqa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coqa_qa_studies": _bench_coqa_qa_studies(seed)}
