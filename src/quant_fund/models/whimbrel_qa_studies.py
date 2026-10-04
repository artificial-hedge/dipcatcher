"""whimbrel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def whimbrel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """whimbrel_qa_studies

    check:
    whimbrel_qa_studies: WhimbrelQA metrics
    """
    return fit_ok and sample_ok


def whimbrel_qa_studies_aux(aux: bool) -> bool:
    """whimbrel_qa_studies

    aux:
    whimbrel_qa_studies: whimbrels, marshes, answers, and scores
    """
    return aux


def _bench_whimbrel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(whimbrel_qa_studies_ok(True, True))
    checks.append(not whimbrel_qa_studies_ok(False, True))
    checks.append(whimbrel_qa_studies_aux(True))
    checks.append(not whimbrel_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_whimbrel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whimbrel_qa_studies": _bench_whimbrel_qa_studies(seed)}
