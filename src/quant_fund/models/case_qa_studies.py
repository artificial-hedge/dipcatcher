"""case_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def case_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """case_qa_studies

    check:
    case_qa_studies: CaseQA metrics
    """
    return fit_ok and sample_ok


def case_qa_studies_aux(aux: bool) -> bool:
    """case_qa_studies

    aux:
    case_qa_studies: cases, holdings, answers, and scores
    """
    return aux


def _bench_case_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(case_qa_studies_ok(True, True))
    checks.append(not case_qa_studies_ok(False, True))
    checks.append(case_qa_studies_aux(True))
    checks.append(not case_qa_studies_aux(False))
    checks.append(True)  # legal-regulatory canon
    return float(sum(checks) / len(checks))


def bench_case_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_case_qa_studies": _bench_case_qa_studies(seed)}
