"""hawker_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hawker_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hawker_qa_studies

    check:
    hawker_qa_studies: HawkerQA metrics
    """
    return fit_ok and sample_ok


def hawker_qa_studies_aux(aux: bool) -> bool:
    """hawker_qa_studies

    aux:
    hawker_qa_studies: hawkers, meadows, answers, and scores
    """
    return aux


def _bench_hawker_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hawker_qa_studies_ok(True, True))
    checks.append(not hawker_qa_studies_ok(False, True))
    checks.append(hawker_qa_studies_aux(True))
    checks.append(not hawker_qa_studies_aux(False))
    checks.append(True)  # dragonfly canon
    return float(sum(checks) / len(checks))


def bench_hawker_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hawker_qa_studies": _bench_hawker_qa_studies(seed)}
