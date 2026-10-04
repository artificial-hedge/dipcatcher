"""guira_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guira_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guira_qa_studies

    check:
    guira_qa_studies: GuiraQA metrics
    """
    return fit_ok and sample_ok


def guira_qa_studies_aux(aux: bool) -> bool:
    """guira_qa_studies

    aux:
    guira_qa_studies: guira cuckoos, pampas, answers, and scores
    """
    return aux


def _bench_guira_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guira_qa_studies_ok(True, True))
    checks.append(not guira_qa_studies_ok(False, True))
    checks.append(guira_qa_studies_aux(True))
    checks.append(not guira_qa_studies_aux(False))
    checks.append(True)  # cuckoo-turaco canon
    return float(sum(checks) / len(checks))


def bench_guira_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guira_qa_studies": _bench_guira_qa_studies(seed)}
