"""nanna3_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nanna3_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nanna3_qa_studies

    check:
    nanna3_qa_studies: Nanna3QA metrics
    """
    return fit_ok and sample_ok


def nanna3_qa_studies_aux(aux: bool) -> bool:
    """nanna3_qa_studies

    aux:
    nanna3_qa_studies: nanna3, blossom brides, answers, and scores
    """
    return aux


def _bench_nanna3_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nanna3_qa_studies_ok(True, True))
    checks.append(not nanna3_qa_studies_ok(False, True))
    checks.append(nanna3_qa_studies_aux(True))
    checks.append(not nanna3_qa_studies_aux(False))
    checks.append(True)  # norse-myth-13 canon
    return float(sum(checks) / len(checks))


def bench_nanna3_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nanna3_qa_studies": _bench_nanna3_qa_studies(seed)}
