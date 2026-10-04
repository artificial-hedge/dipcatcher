"""inkcap_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def inkcap_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inkcap_qa_studies

    check:
    inkcap_qa_studies: InkcapQA metrics
    """
    return fit_ok and sample_ok


def inkcap_qa_studies_aux(aux: bool) -> bool:
    """inkcap_qa_studies

    aux:
    inkcap_qa_studies: inkcaps, composts, answers, and scores
    """
    return aux


def _bench_inkcap_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(inkcap_qa_studies_ok(True, True))
    checks.append(not inkcap_qa_studies_ok(False, True))
    checks.append(inkcap_qa_studies_aux(True))
    checks.append(not inkcap_qa_studies_aux(False))
    checks.append(True)  # fungi canon
    return float(sum(checks) / len(checks))


def bench_inkcap_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inkcap_qa_studies": _bench_inkcap_qa_studies(seed)}
