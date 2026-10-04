"""panther_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def panther_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """panther_qa_studies

    check:
    panther_qa_studies: PantherQA metrics
    """
    return fit_ok and sample_ok


def panther_qa_studies_aux(aux: bool) -> bool:
    """panther_qa_studies

    aux:
    panther_qa_studies: panthers, shadows, answers, and scores
    """
    return aux


def _bench_panther_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(panther_qa_studies_ok(True, True))
    checks.append(not panther_qa_studies_ok(False, True))
    checks.append(panther_qa_studies_aux(True))
    checks.append(not panther_qa_studies_aux(False))
    checks.append(True)  # wildcat-2 canon
    return float(sum(checks) / len(checks))


def bench_panther_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_panther_qa_studies": _bench_panther_qa_studies(seed)}
