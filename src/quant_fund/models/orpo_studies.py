"""orpo_studies module (SYNTHETIC)."""

from __future__ import annotations


def orpo_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """orpo_studies

    check:
    orpo_studies: odds ratio and reference-free/preference and penalties
    """
    return fit_ok and sample_ok


def orpo_studies_aux(aux: bool) -> bool:
    """orpo_studies

    aux:
    orpo_studies: monolithic loss and discrimination/odds and updates
    """
    return aux


def _bench_orpo_studies(seed: int = 0) -> float:
    checks = []
    checks.append(orpo_studies_ok(True, True))
    checks.append(not orpo_studies_ok(False, True))
    checks.append(orpo_studies_aux(True))
    checks.append(not orpo_studies_aux(False))
    checks.append(True)  # post-training-2 canon
    return float(sum(checks) / len(checks))


def bench_orpo_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orpo_studies": _bench_orpo_studies(seed)}
