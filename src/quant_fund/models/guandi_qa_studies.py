"""guandi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guandi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guandi_qa_studies

    check:
    guandi_qa_studies: GuandiQA metrics
    """
    return fit_ok and sample_ok


def guandi_qa_studies_aux(aux: bool) -> bool:
    """guandi_qa_studies

    aux:
    guandi_qa_studies: guandi, war gods, answers, and scores
    """
    return aux


def _bench_guandi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guandi_qa_studies_ok(True, True))
    checks.append(not guandi_qa_studies_ok(False, True))
    checks.append(guandi_qa_studies_aux(True))
    checks.append(not guandi_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_guandi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guandi_qa_studies": _bench_guandi_qa_studies(seed)}
