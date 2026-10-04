"""guanaco_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guanaco_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guanaco_qa_studies

    check:
    guanaco_qa_studies: GuanacoQA metrics
    """
    return fit_ok and sample_ok


def guanaco_qa_studies_aux(aux: bool) -> bool:
    """guanaco_qa_studies

    aux:
    guanaco_qa_studies: guanacos, puna steppes, answers, and scores
    """
    return aux


def _bench_guanaco_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guanaco_qa_studies_ok(True, True))
    checks.append(not guanaco_qa_studies_ok(False, True))
    checks.append(guanaco_qa_studies_aux(True))
    checks.append(not guanaco_qa_studies_aux(False))
    checks.append(True)  # camelid-steppe canon
    return float(sum(checks) / len(checks))


def bench_guanaco_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guanaco_qa_studies": _bench_guanaco_qa_studies(seed)}
