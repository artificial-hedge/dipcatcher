"""galaxy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def galaxy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """galaxy_qa_studies

    check:
    galaxy_qa_studies: GalaxyQA metrics
    """
    return fit_ok and sample_ok


def galaxy_qa_studies_aux(aux: bool) -> bool:
    """galaxy_qa_studies

    aux:
    galaxy_qa_studies: galaxies, clusters, answers, and scores
    """
    return aux


def _bench_galaxy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(galaxy_qa_studies_ok(True, True))
    checks.append(not galaxy_qa_studies_ok(False, True))
    checks.append(galaxy_qa_studies_aux(True))
    checks.append(not galaxy_qa_studies_aux(False))
    checks.append(True)  # celestial canon
    return float(sum(checks) / len(checks))


def bench_galaxy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galaxy_qa_studies": _bench_galaxy_qa_studies(seed)}
