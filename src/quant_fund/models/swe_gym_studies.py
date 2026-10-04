"""swe_gym_studies module (SYNTHETIC)."""

from __future__ import annotations


def swe_gym_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swe_gym_studies

    check:
    swe_gym_studies: SWE-Gym metrics
    """
    return fit_ok and sample_ok


def swe_gym_studies_aux(aux: bool) -> bool:
    """swe_gym_studies

    aux:
    swe_gym_studies: issues, patches, tests, and scores
    """
    return aux


def _bench_swe_gym_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swe_gym_studies_ok(True, True))
    checks.append(not swe_gym_studies_ok(False, True))
    checks.append(swe_gym_studies_aux(True))
    checks.append(not swe_gym_studies_aux(False))
    checks.append(True)  # code-agent canon
    return float(sum(checks) / len(checks))


def bench_swe_gym_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swe_gym_studies": _bench_swe_gym_studies(seed)}
