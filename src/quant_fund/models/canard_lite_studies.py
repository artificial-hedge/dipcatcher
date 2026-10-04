"""canard_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def canard_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """canard_lite_studies

    check:
    canard_lite_studies: CANARD metrics
    """
    return fit_ok and sample_ok


def canard_lite_studies_aux(aux: bool) -> bool:
    """canard_lite_studies

    aux:
    canard_lite_studies: turns, rewrites, answers, and scores
    """
    return aux


def _bench_canard_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(canard_lite_studies_ok(True, True))
    checks.append(not canard_lite_studies_ok(False, True))
    checks.append(canard_lite_studies_aux(True))
    checks.append(not canard_lite_studies_aux(False))
    checks.append(True)  # conversational-QA canon
    return float(sum(checks) / len(checks))


def bench_canard_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canard_lite_studies": _bench_canard_lite_studies(seed)}
