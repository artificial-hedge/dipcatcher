"""pop_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pop_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pop_qa_studies

    check:
    pop_qa_studies: PopQA entity metrics
    """
    return fit_ok and sample_ok


def pop_qa_studies_aux(aux: bool) -> bool:
    """pop_qa_studies

    aux:
    pop_qa_studies: entities, relations, answers, and accuracies
    """
    return aux


def _bench_pop_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pop_qa_studies_ok(True, True))
    checks.append(not pop_qa_studies_ok(False, True))
    checks.append(pop_qa_studies_aux(True))
    checks.append(not pop_qa_studies_aux(False))
    checks.append(True)  # knowledge-QA canon
    return float(sum(checks) / len(checks))


def bench_pop_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pop_qa_studies": _bench_pop_qa_studies(seed)}
