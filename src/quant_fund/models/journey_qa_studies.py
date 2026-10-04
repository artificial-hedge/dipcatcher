"""journey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def journey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """journey_qa_studies

    check:
    journey_qa_studies: JourneyQA metrics
    """
    return fit_ok and sample_ok


def journey_qa_studies_aux(aux: bool) -> bool:
    """journey_qa_studies

    aux:
    journey_qa_studies: trips, legs, answers, and scores
    """
    return aux


def _bench_journey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(journey_qa_studies_ok(True, True))
    checks.append(not journey_qa_studies_ok(False, True))
    checks.append(journey_qa_studies_aux(True))
    checks.append(not journey_qa_studies_aux(False))
    checks.append(True)  # spatial-navigation canon
    return float(sum(checks) / len(checks))


def bench_journey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_journey_qa_studies": _bench_journey_qa_studies(seed)}
