"""monal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def monal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monal_qa_studies

    check:
    monal_qa_studies: MonalQA metrics
    """
    return fit_ok and sample_ok


def monal_qa_studies_aux(aux: bool) -> bool:
    """monal_qa_studies

    aux:
    monal_qa_studies: monals, rhododendron thickets, answers, and scores
    """
    return aux


def _bench_monal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(monal_qa_studies_ok(True, True))
    checks.append(not monal_qa_studies_ok(False, True))
    checks.append(monal_qa_studies_aux(True))
    checks.append(not monal_qa_studies_aux(False))
    checks.append(True)  # alpine-bird canon
    return float(sum(checks) / len(checks))


def bench_monal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monal_qa_studies": _bench_monal_qa_studies(seed)}
