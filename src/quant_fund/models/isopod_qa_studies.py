"""isopod_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def isopod_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """isopod_qa_studies

    check:
    isopod_qa_studies: IsopodQA metrics
    """
    return fit_ok and sample_ok


def isopod_qa_studies_aux(aux: bool) -> bool:
    """isopod_qa_studies

    aux:
    isopod_qa_studies: isopods, wrack lines, answers, and scores
    """
    return aux


def _bench_isopod_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(isopod_qa_studies_ok(True, True))
    checks.append(not isopod_qa_studies_ok(False, True))
    checks.append(isopod_qa_studies_aux(True))
    checks.append(not isopod_qa_studies_aux(False))
    checks.append(True)  # plankton-shore canon
    return float(sum(checks) / len(checks))


def bench_isopod_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isopod_qa_studies": _bench_isopod_qa_studies(seed)}
