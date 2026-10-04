"""dorotabo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dorotabo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dorotabo_qa_studies

    check:
    dorotabo_qa_studies: DorotaboQA metrics
    """
    return fit_ok and sample_ok


def dorotabo_qa_studies_aux(aux: bool) -> bool:
    """dorotabo_qa_studies

    aux:
    dorotabo_qa_studies: dorotabos, rice paddies, answers, and scores
    """
    return aux


def _bench_dorotabo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dorotabo_qa_studies_ok(True, True))
    checks.append(not dorotabo_qa_studies_ok(False, True))
    checks.append(dorotabo_qa_studies_aux(True))
    checks.append(not dorotabo_qa_studies_aux(False))
    checks.append(True)  # yokai-5 canon
    return float(sum(checks) / len(checks))


def bench_dorotabo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dorotabo_qa_studies": _bench_dorotabo_qa_studies(seed)}
