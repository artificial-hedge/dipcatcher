"""rush_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rush_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rush_qa_studies

    check:
    rush_qa_studies: RushQA metrics
    """
    return fit_ok and sample_ok


def rush_qa_studies_aux(aux: bool) -> bool:
    """rush_qa_studies

    aux:
    rush_qa_studies: rushes, ditches, answers, and scores
    """
    return aux


def _bench_rush_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rush_qa_studies_ok(True, True))
    checks.append(not rush_qa_studies_ok(False, True))
    checks.append(rush_qa_studies_aux(True))
    checks.append(not rush_qa_studies_aux(False))
    checks.append(True)  # sedge canon
    return float(sum(checks) / len(checks))


def bench_rush_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rush_qa_studies": _bench_rush_qa_studies(seed)}
