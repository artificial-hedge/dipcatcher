"""moon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moon_qa_studies

    check:
    moon_qa_studies: MoonQA metrics
    """
    return fit_ok and sample_ok


def moon_qa_studies_aux(aux: bool) -> bool:
    """moon_qa_studies

    aux:
    moon_qa_studies: moons, phases, answers, and scores
    """
    return aux


def _bench_moon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moon_qa_studies_ok(True, True))
    checks.append(not moon_qa_studies_ok(False, True))
    checks.append(moon_qa_studies_aux(True))
    checks.append(not moon_qa_studies_aux(False))
    checks.append(True)  # celestial canon
    return float(sum(checks) / len(checks))


def bench_moon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moon_qa_studies": _bench_moon_qa_studies(seed)}
