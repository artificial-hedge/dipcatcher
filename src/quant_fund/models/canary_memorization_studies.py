"""canary_memorization_studies module (SYNTHETIC)."""

from __future__ import annotations


def canary_memorization_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """canary_memorization_studies

    check:
    canary_memorization_studies: Canary memorization exposure metrics
    """
    return fit_ok and sample_ok


def canary_memorization_studies_aux(aux: bool) -> bool:
    """canary_memorization_studies

    aux:
    canary_memorization_studies: canaries, exposure, ranks, and rates
    """
    return aux


def _bench_canary_memorization_studies(seed: int = 0) -> float:
    checks = []
    checks.append(canary_memorization_studies_ok(True, True))
    checks.append(not canary_memorization_studies_ok(False, True))
    checks.append(canary_memorization_studies_aux(True))
    checks.append(not canary_memorization_studies_aux(False))
    checks.append(True)  # privacy-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_canary_memorization_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canary_memorization_studies": _bench_canary_memorization_studies(seed)}
