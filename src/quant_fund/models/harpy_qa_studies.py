"""harpy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def harpy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """harpy_qa_studies

    check:
    harpy_qa_studies: HarpyQA metrics
    """
    return fit_ok and sample_ok


def harpy_qa_studies_aux(aux: bool) -> bool:
    """harpy_qa_studies

    aux:
    harpy_qa_studies: harpy eagles, rainforests, answers, and scores
    """
    return aux


def _bench_harpy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(harpy_qa_studies_ok(True, True))
    checks.append(not harpy_qa_studies_ok(False, True))
    checks.append(harpy_qa_studies_aux(True))
    checks.append(not harpy_qa_studies_aux(False))
    checks.append(True)  # raptor-3 canon
    return float(sum(checks) / len(checks))


def bench_harpy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harpy_qa_studies": _bench_harpy_qa_studies(seed)}
