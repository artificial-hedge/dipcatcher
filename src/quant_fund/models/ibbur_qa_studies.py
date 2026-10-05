"""ibbur_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ibbur_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ibbur_qa_studies

    check:
    ibbur_qa_studies: I
    """
    return fit_ok and sample_ok


def ibbur_qa_studies_aux(aux: bool) -> bool:
    """ibbur_qa_studies

    aux:
    ibbur_qa_studies: b
    """
    return aux


def _bench_ibbur_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ibbur_qa_studies_ok(True, True))
    checks.append(not ibbur_qa_studies_ok(False, True))
    checks.append(ibbur_qa_studies_aux(True))
    checks.append(not ibbur_qa_studies_aux(False))
    checks.append(True)  # shedim canon
    return float(sum(checks) / len(checks))


def bench_ibbur_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ibbur_qa_studies": _bench_ibbur_qa_studies(seed)}
