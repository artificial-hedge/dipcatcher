"""thornback_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def thornback_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thornback_qa_studies

    check:
    thornback_qa_studies: ThornbackQA metrics
    """
    return fit_ok and sample_ok


def thornback_qa_studies_aux(aux: bool) -> bool:
    """thornback_qa_studies

    aux:
    thornback_qa_studies: thornback rays, cold shelves, answers, and scores
    """
    return aux


def _bench_thornback_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(thornback_qa_studies_ok(True, True))
    checks.append(not thornback_qa_studies_ok(False, True))
    checks.append(thornback_qa_studies_aux(True))
    checks.append(not thornback_qa_studies_aux(False))
    checks.append(True)  # ray canon
    return float(sum(checks) / len(checks))


def bench_thornback_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thornback_qa_studies": _bench_thornback_qa_studies(seed)}
