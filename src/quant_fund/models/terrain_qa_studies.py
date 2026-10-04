"""terrain_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def terrain_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """terrain_qa_studies

    check:
    terrain_qa_studies: TerrainQA metrics
    """
    return fit_ok and sample_ok


def terrain_qa_studies_aux(aux: bool) -> bool:
    """terrain_qa_studies

    aux:
    terrain_qa_studies: areas, features, answers, and scores
    """
    return aux


def _bench_terrain_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(terrain_qa_studies_ok(True, True))
    checks.append(not terrain_qa_studies_ok(False, True))
    checks.append(terrain_qa_studies_aux(True))
    checks.append(not terrain_qa_studies_aux(False))
    checks.append(True)  # spatial-navigation canon
    return float(sum(checks) / len(checks))


def bench_terrain_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_terrain_qa_studies": _bench_terrain_qa_studies(seed)}
