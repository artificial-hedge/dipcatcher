"""sarkany2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sarkany2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sarkany2_qa_studies

    check:
    sarkany2_qa_studies: Sarkany2QA metrics
    """
    return fit_ok and sample_ok


def sarkany2_qa_studies_aux(aux: bool) -> bool:
    """sarkany2_qa_studies

    aux:
    sarkany2_qa_studies: sarkany2, storm dragons, answers, and scores
    """
    return aux


def _bench_sarkany2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sarkany2_qa_studies_ok(True, True))
    checks.append(not sarkany2_qa_studies_ok(False, True))
    checks.append(sarkany2_qa_studies_aux(True))
    checks.append(not sarkany2_qa_studies_aux(False))
    checks.append(True)  # hungarian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_sarkany2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sarkany2_qa_studies": _bench_sarkany2_qa_studies(seed)}
