"""rail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rail_qa_studies

    check:
    rail_qa_studies: RailQA metrics
    """
    return fit_ok and sample_ok


def rail_qa_studies_aux(aux: bool) -> bool:
    """rail_qa_studies

    aux:
    rail_qa_studies: rails, marshes, answers, and scores
    """
    return aux


def _bench_rail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rail_qa_studies_ok(True, True))
    checks.append(not rail_qa_studies_ok(False, True))
    checks.append(rail_qa_studies_aux(True))
    checks.append(not rail_qa_studies_aux(False))
    checks.append(True)  # marshbird canon
    return float(sum(checks) / len(checks))


def bench_rail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rail_qa_studies": _bench_rail_qa_studies(seed)}
