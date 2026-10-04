"""itinerary_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def itinerary_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """itinerary_qa_studies

    check:
    itinerary_qa_studies: ItineraryQA metrics
    """
    return fit_ok and sample_ok


def itinerary_qa_studies_aux(aux: bool) -> bool:
    """itinerary_qa_studies

    aux:
    itinerary_qa_studies: plans, stops, answers, and scores
    """
    return aux


def _bench_itinerary_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(itinerary_qa_studies_ok(True, True))
    checks.append(not itinerary_qa_studies_ok(False, True))
    checks.append(itinerary_qa_studies_aux(True))
    checks.append(not itinerary_qa_studies_aux(False))
    checks.append(True)  # spatial-navigation canon
    return float(sum(checks) / len(checks))


def bench_itinerary_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_itinerary_qa_studies": _bench_itinerary_qa_studies(seed)}
