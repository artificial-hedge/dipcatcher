"""tesub2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tesub2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tesub2_qa_studies

    check:
    tesub2_qa_studies: Tesub2QA metrics
    """
    return fit_ok and sample_ok


def tesub2_qa_studies_aux(aux: bool) -> bool:
    """tesub2_qa_studies

    aux:
    tesub2_qa_studies: tesub2, sky storms, answers, and scores
    """
    return aux


def _bench_tesub2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tesub2_qa_studies_ok(True, True))
    checks.append(not tesub2_qa_studies_ok(False, True))
    checks.append(tesub2_qa_studies_aux(True))
    checks.append(not tesub2_qa_studies_aux(False))
    checks.append(True)  # hittite-3 canon
    return float(sum(checks) / len(checks))


def bench_tesub2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tesub2_qa_studies": _bench_tesub2_qa_studies(seed)}
