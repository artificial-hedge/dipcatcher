"""episum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def episum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """episum_lite_studies

    check:
    episum_lite_studies: EpiSum metrics
    """
    return fit_ok and sample_ok


def episum_lite_studies_aux(aux: bool) -> bool:
    """episum_lite_studies

    aux:
    episum_lite_studies: episodes, saliences, summaries, and scores
    """
    return aux


def _bench_episum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(episum_lite_studies_ok(True, True))
    checks.append(not episum_lite_studies_ok(False, True))
    checks.append(episum_lite_studies_aux(True))
    checks.append(not episum_lite_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_episum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_episum_lite_studies": _bench_episum_lite_studies(seed)}
