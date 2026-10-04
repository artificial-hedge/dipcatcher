"""simple_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def simple_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """simple_qa_studies

    check:
    simple_qa_studies: SimpleQA metrics
    """
    return fit_ok and sample_ok


def simple_qa_studies_aux(aux: bool) -> bool:
    """simple_qa_studies

    aux:
    simple_qa_studies: questions, answers, entities, and scores
    """
    return aux


def _bench_simple_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(simple_qa_studies_ok(True, True))
    checks.append(not simple_qa_studies_ok(False, True))
    checks.append(simple_qa_studies_aux(True))
    checks.append(not simple_qa_studies_aux(False))
    checks.append(True)  # multilingual-QA canon
    return float(sum(checks) / len(checks))


def bench_simple_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simple_qa_studies": _bench_simple_qa_studies(seed)}
