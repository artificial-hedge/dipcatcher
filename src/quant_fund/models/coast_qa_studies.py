"""coast_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def coast_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coast_qa_studies

    check:
    coast_qa_studies: CoastQA metrics
    """
    return fit_ok and sample_ok


def coast_qa_studies_aux(aux: bool) -> bool:
    """coast_qa_studies

    aux:
    coast_qa_studies: coasts, landmarks, answers, and scores
    """
    return aux


def _bench_coast_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(coast_qa_studies_ok(True, True))
    checks.append(not coast_qa_studies_ok(False, True))
    checks.append(coast_qa_studies_aux(True))
    checks.append(not coast_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_coast_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coast_qa_studies": _bench_coast_qa_studies(seed)}
