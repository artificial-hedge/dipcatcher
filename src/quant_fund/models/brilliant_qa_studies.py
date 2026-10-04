"""brilliant_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def brilliant_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brilliant_qa_studies

    check:
    brilliant_qa_studies: BrilliantQA metrics
    """
    return fit_ok and sample_ok


def brilliant_qa_studies_aux(aux: bool) -> bool:
    """brilliant_qa_studies

    aux:
    brilliant_qa_studies: brilliants, cloudforests, answers, and scores
    """
    return aux


def _bench_brilliant_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(brilliant_qa_studies_ok(True, True))
    checks.append(not brilliant_qa_studies_ok(False, True))
    checks.append(brilliant_qa_studies_aux(True))
    checks.append(not brilliant_qa_studies_aux(False))
    checks.append(True)  # hummingbird canon
    return float(sum(checks) / len(checks))


def bench_brilliant_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brilliant_qa_studies": _bench_brilliant_qa_studies(seed)}
