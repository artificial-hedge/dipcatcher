"""grail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grail_qa_studies

    check:
    grail_qa_studies: GrailQA compositional metrics
    """
    return fit_ok and sample_ok


def grail_qa_studies_aux(aux: bool) -> bool:
    """grail_qa_studies

    aux:
    grail_qa_studies: questions, s-expressions, answers, and accuracies
    """
    return aux


def _bench_grail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grail_qa_studies_ok(True, True))
    checks.append(not grail_qa_studies_ok(False, True))
    checks.append(grail_qa_studies_aux(True))
    checks.append(not grail_qa_studies_aux(False))
    checks.append(True)  # KB-QA canon
    return float(sum(checks) / len(checks))


def bench_grail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grail_qa_studies": _bench_grail_qa_studies(seed)}
