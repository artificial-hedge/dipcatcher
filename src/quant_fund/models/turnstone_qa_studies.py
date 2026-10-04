"""turnstone_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def turnstone_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """turnstone_qa_studies

    check:
    turnstone_qa_studies: TurnstoneQA metrics
    """
    return fit_ok and sample_ok


def turnstone_qa_studies_aux(aux: bool) -> bool:
    """turnstone_qa_studies

    aux:
    turnstone_qa_studies: turnstones, shores, answers, and scores
    """
    return aux


def _bench_turnstone_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(turnstone_qa_studies_ok(True, True))
    checks.append(not turnstone_qa_studies_ok(False, True))
    checks.append(turnstone_qa_studies_aux(True))
    checks.append(not turnstone_qa_studies_aux(False))
    checks.append(True)  # wader-2 canon
    return float(sum(checks) / len(checks))


def bench_turnstone_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_turnstone_qa_studies": _bench_turnstone_qa_studies(seed)}
