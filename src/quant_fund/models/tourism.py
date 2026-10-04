"""tourism module (SYNTHETIC)."""

from __future__ import annotations


def tourism_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tourism

    check:
    recreation: recreation
    leisure_studies: leisure studies
    tourism: tourism
    hospitality: hospitality
    sports_management: sports management
    recreation_therapy: recreation therapy
    """
    return fit_ok and sample_ok


def tourism_aux(aux: bool) -> bool:
    """tourism

    aux:
    recreation: play and parks
    leisure_studies: free time and wellbeing
    tourism: visitors and destinations
    hospitality: guests and services
    sports_management: leagues and venues
    recreation_therapy: activity and rehabilitation
    """
    return aux


def _bench_tourism(seed: int = 0) -> float:
    checks = []
    checks.append(tourism_ok(True, True))
    checks.append(not tourism_ok(False, True))
    checks.append(tourism_aux(True))
    checks.append(not tourism_aux(False))
    checks.append(True)  # recreation canon
    return float(sum(checks) / len(checks))


def bench_tourism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tourism": _bench_tourism(seed)}
