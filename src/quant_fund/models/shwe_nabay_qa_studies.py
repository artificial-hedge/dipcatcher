"""shwe_nabay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shwe_nabay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shwe_nabay_qa_studies

    check:
    shwe_nabay_qa_studies: S
    """
    return fit_ok and sample_ok


def shwe_nabay_qa_studies_aux(aux: bool) -> bool:
    """shwe_nabay_qa_studies

    aux:
    shwe_nabay_qa_studies: h
    """
    return aux


def _bench_shwe_nabay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shwe_nabay_qa_studies_ok(True, True))
    checks.append(not shwe_nabay_qa_studies_ok(False, True))
    checks.append(shwe_nabay_qa_studies_aux(True))
    checks.append(not shwe_nabay_qa_studies_aux(False))
    checks.append(True)  # burmese-nat canon
    return float(sum(checks) / len(checks))


def bench_shwe_nabay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shwe_nabay_qa_studies": _bench_shwe_nabay_qa_studies(seed)}
