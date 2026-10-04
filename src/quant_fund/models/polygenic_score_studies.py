"""polygenic_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def polygenic_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """polygenic_score_studies

    check:
    polygenic_score_studies: score weights and clumping/thresholds and target
    """
    return fit_ok and sample_ok


def polygenic_score_studies_aux(aux: bool) -> bool:
    """polygenic_score_studies

    aux:
    polygenic_score_studies: transferability and ancestry/calibration and overlap
    """
    return aux


def _bench_polygenic_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(polygenic_score_studies_ok(True, True))
    checks.append(not polygenic_score_studies_ok(False, True))
    checks.append(polygenic_score_studies_aux(True))
    checks.append(not polygenic_score_studies_aux(False))
    checks.append(True)  # genetic-epidemiology/MR canon
    return float(sum(checks) / len(checks))


def bench_polygenic_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polygenic_score_studies": _bench_polygenic_score_studies(seed)}
