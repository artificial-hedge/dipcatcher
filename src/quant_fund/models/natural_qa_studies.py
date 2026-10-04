"""natural_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def natural_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """natural_qa_studies

    check:
    natural_qa_studies: Natural-Questions metrics
    """
    return fit_ok and sample_ok


def natural_qa_studies_aux(aux: bool) -> bool:
    """natural_qa_studies

    aux:
    natural_qa_studies: questions, answers, contexts, and accuracies
    """
    return aux


def _bench_natural_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(natural_qa_studies_ok(True, True))
    checks.append(not natural_qa_studies_ok(False, True))
    checks.append(natural_qa_studies_aux(True))
    checks.append(not natural_qa_studies_aux(False))
    checks.append(True)  # knowledge-QA canon
    return float(sum(checks) / len(checks))


def bench_natural_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_natural_qa_studies": _bench_natural_qa_studies(seed)}
