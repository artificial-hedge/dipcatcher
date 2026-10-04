"""web_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def web_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """web_qa_studies

    check:
    web_qa_studies: WebQuestions factoid metrics
    """
    return fit_ok and sample_ok


def web_qa_studies_aux(aux: bool) -> bool:
    """web_qa_studies

    aux:
    web_qa_studies: questions, entities, answers, and accuracies
    """
    return aux


def _bench_web_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(web_qa_studies_ok(True, True))
    checks.append(not web_qa_studies_ok(False, True))
    checks.append(web_qa_studies_aux(True))
    checks.append(not web_qa_studies_aux(False))
    checks.append(True)  # open-domain-QA canon
    return float(sum(checks) / len(checks))


def bench_web_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_web_qa_studies": _bench_web_qa_studies(seed)}
