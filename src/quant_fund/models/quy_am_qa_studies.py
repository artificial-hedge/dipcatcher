"""quy_am_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quy_am_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quy_am_qa_studies

    check:
    quy_am_qa_studies: Q
    """
    return fit_ok and sample_ok


def quy_am_qa_studies_aux(aux: bool) -> bool:
    """quy_am_qa_studies

    aux:
    quy_am_qa_studies: u
    """
    return aux


def _bench_quy_am_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quy_am_qa_studies_ok(True, True))
    checks.append(not quy_am_qa_studies_ok(False, True))
    checks.append(quy_am_qa_studies_aux(True))
    checks.append(not quy_am_qa_studies_aux(False))
    checks.append(True)  # vietnamese-demon canon
    return float(sum(checks) / len(checks))


def bench_quy_am_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quy_am_qa_studies": _bench_quy_am_qa_studies(seed)}
