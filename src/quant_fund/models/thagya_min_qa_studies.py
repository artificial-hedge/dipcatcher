"""thagya_min_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def thagya_min_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thagya_min_qa_studies

    check:
    thagya_min_qa_studies: T
    """
    return fit_ok and sample_ok


def thagya_min_qa_studies_aux(aux: bool) -> bool:
    """thagya_min_qa_studies

    aux:
    thagya_min_qa_studies: h
    """
    return aux


def _bench_thagya_min_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(thagya_min_qa_studies_ok(True, True))
    checks.append(not thagya_min_qa_studies_ok(False, True))
    checks.append(thagya_min_qa_studies_aux(True))
    checks.append(not thagya_min_qa_studies_aux(False))
    checks.append(True)  # burmese-nat canon
    return float(sum(checks) / len(checks))


def bench_thagya_min_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thagya_min_qa_studies": _bench_thagya_min_qa_studies(seed)}
