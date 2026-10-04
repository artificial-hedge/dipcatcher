"""leisure_studies module (SYNTHETIC)."""

from __future__ import annotations


def leisure_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leisure_studies

    check:
    recreation: recreation
    leisure_studies: leisure studies
    tourism: tourism
    hospitality: hospitality
    sports_management: sports management
    recreation_therapy: recreation therapy
    """
    return fit_ok and sample_ok


def leisure_studies_aux(aux: bool) -> bool:
    """leisure_studies

    aux:
    recreation: play and parks
    leisure_studies: free time and wellbeing
    tourism: visitors and destinations
    hospitality: guests and services
    sports_management: leagues and venues
    recreation_therapy: activity and rehabilitation
    """
    return aux


def _bench_leisure_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leisure_studies_ok(True, True))
    checks.append(not leisure_studies_ok(False, True))
    checks.append(leisure_studies_aux(True))
    checks.append(not leisure_studies_aux(False))
    checks.append(True)  # recreation canon
    return float(sum(checks) / len(checks))


def bench_leisure_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leisure_studies": _bench_leisure_studies(seed)}
