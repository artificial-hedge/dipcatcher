"""roller_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def roller_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """roller_qa_studies

    check:
    roller_qa_studies: RollerQA metrics
    """
    return fit_ok and sample_ok


def roller_qa_studies_aux(aux: bool) -> bool:
    """roller_qa_studies

    aux:
    roller_qa_studies: rollers, grasslands, answers, and scores
    """
    return aux


def _bench_roller_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(roller_qa_studies_ok(True, True))
    checks.append(not roller_qa_studies_ok(False, True))
    checks.append(roller_qa_studies_aux(True))
    checks.append(not roller_qa_studies_aux(False))
    checks.append(True)  # riverbird canon
    return float(sum(checks) / len(checks))


def bench_roller_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roller_qa_studies": _bench_roller_qa_studies(seed)}
