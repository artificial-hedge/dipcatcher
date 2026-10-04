"""aircraft_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aircraft_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aircraft_qa_studies

    check:
    aircraft_qa_studies: AircraftQA metrics
    """
    return fit_ok and sample_ok


def aircraft_qa_studies_aux(aux: bool) -> bool:
    """aircraft_qa_studies

    aux:
    aircraft_qa_studies: aircraft, systems, answers, and scores
    """
    return aux


def _bench_aircraft_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aircraft_qa_studies_ok(True, True))
    checks.append(not aircraft_qa_studies_ok(False, True))
    checks.append(aircraft_qa_studies_aux(True))
    checks.append(not aircraft_qa_studies_aux(False))
    checks.append(True)  # vehicle canon
    return float(sum(checks) / len(checks))


def bench_aircraft_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aircraft_qa_studies": _bench_aircraft_qa_studies(seed)}
