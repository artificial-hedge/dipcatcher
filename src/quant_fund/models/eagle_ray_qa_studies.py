"""eagle_ray_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def eagle_ray_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """eagle_ray_qa_studies

    check:
    eagle_ray_qa_studies: EagleRayQA metrics
    """
    return fit_ok and sample_ok


def eagle_ray_qa_studies_aux(aux: bool) -> bool:
    """eagle_ray_qa_studies

    aux:
    eagle_ray_qa_studies: eagle rays, reef flats, answers, and scores
    """
    return aux


def _bench_eagle_ray_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(eagle_ray_qa_studies_ok(True, True))
    checks.append(not eagle_ray_qa_studies_ok(False, True))
    checks.append(eagle_ray_qa_studies_aux(True))
    checks.append(not eagle_ray_qa_studies_aux(False))
    checks.append(True)  # ray canon
    return float(sum(checks) / len(checks))


def bench_eagle_ray_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eagle_ray_qa_studies": _bench_eagle_ray_qa_studies(seed)}
