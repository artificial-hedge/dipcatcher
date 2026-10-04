"""babyai_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def babyai_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """babyai_lite_studies

    check:
    babyai_lite_studies: BabyAI metrics
    """
    return fit_ok and sample_ok


def babyai_lite_studies_aux(aux: bool) -> bool:
    """babyai_lite_studies

    aux:
    babyai_lite_studies: missions, grids, actions, and scores
    """
    return aux


def _bench_babyai_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(babyai_lite_studies_ok(True, True))
    checks.append(not babyai_lite_studies_ok(False, True))
    checks.append(babyai_lite_studies_aux(True))
    checks.append(not babyai_lite_studies_aux(False))
    checks.append(True)  # embodied-game canon
    return float(sum(checks) / len(checks))


def bench_babyai_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_babyai_lite_studies": _bench_babyai_lite_studies(seed)}
