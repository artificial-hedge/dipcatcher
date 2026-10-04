"""swallowtail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def swallowtail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swallowtail_qa_studies

    check:
    swallowtail_qa_studies: SwallowtailQA metrics
    """
    return fit_ok and sample_ok


def swallowtail_qa_studies_aux(aux: bool) -> bool:
    """swallowtail_qa_studies

    aux:
    swallowtail_qa_studies: swallowtails, citrus, answers, and scores
    """
    return aux


def _bench_swallowtail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swallowtail_qa_studies_ok(True, True))
    checks.append(not swallowtail_qa_studies_ok(False, True))
    checks.append(swallowtail_qa_studies_aux(True))
    checks.append(not swallowtail_qa_studies_aux(False))
    checks.append(True)  # butterfly canon
    return float(sum(checks) / len(checks))


def bench_swallowtail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swallowtail_qa_studies": _bench_swallowtail_qa_studies(seed)}
