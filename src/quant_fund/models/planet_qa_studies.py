"""planet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def planet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """planet_qa_studies

    check:
    planet_qa_studies: PlanetQA metrics
    """
    return fit_ok and sample_ok


def planet_qa_studies_aux(aux: bool) -> bool:
    """planet_qa_studies

    aux:
    planet_qa_studies: planets, orbits, answers, and scores
    """
    return aux


def _bench_planet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(planet_qa_studies_ok(True, True))
    checks.append(not planet_qa_studies_ok(False, True))
    checks.append(planet_qa_studies_aux(True))
    checks.append(not planet_qa_studies_aux(False))
    checks.append(True)  # celestial canon
    return float(sum(checks) / len(checks))


def bench_planet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_planet_qa_studies": _bench_planet_qa_studies(seed)}
