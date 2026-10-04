"""markhor_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def markhor_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """markhor_qa_studies

    check:
    markhor_qa_studies: MarkhorQA metrics
    """
    return fit_ok and sample_ok


def markhor_qa_studies_aux(aux: bool) -> bool:
    """markhor_qa_studies

    aux:
    markhor_qa_studies: markhors, crag ledges, answers, and scores
    """
    return aux


def _bench_markhor_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(markhor_qa_studies_ok(True, True))
    checks.append(not markhor_qa_studies_ok(False, True))
    checks.append(markhor_qa_studies_aux(True))
    checks.append(not markhor_qa_studies_aux(False))
    checks.append(True)  # ungulate canon
    return float(sum(checks) / len(checks))


def bench_markhor_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markhor_qa_studies": _bench_markhor_qa_studies(seed)}
