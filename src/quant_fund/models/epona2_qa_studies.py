"""epona2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def epona2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """epona2_qa_studies

    check:
    epona2_qa_studies: Epona2QA metrics
    """
    return fit_ok and sample_ok


def epona2_qa_studies_aux(aux: bool) -> bool:
    """epona2_qa_studies

    aux:
    epona2_qa_studies: epona2, horse mothers, answers, and scores
    """
    return aux


def _bench_epona2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(epona2_qa_studies_ok(True, True))
    checks.append(not epona2_qa_studies_ok(False, True))
    checks.append(epona2_qa_studies_aux(True))
    checks.append(not epona2_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-4 canon
    return float(sum(checks) / len(checks))


def bench_epona2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epona2_qa_studies": _bench_epona2_qa_studies(seed)}
