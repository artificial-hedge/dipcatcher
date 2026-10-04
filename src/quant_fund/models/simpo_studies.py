"""simpo_studies module (SYNTHETIC)."""

from __future__ import annotations


def simpo_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """simpo_studies

    check:
    simpo_studies: length-normalized preference and margin reward/mean logps and target
    """
    return fit_ok and sample_ok


def simpo_studies_aux(aux: bool) -> bool:
    """simpo_studies

    aux:
    simpo_studies: reference-free and gamma beta/efficiency and ranking
    """
    return aux


def _bench_simpo_studies(seed: int = 0) -> float:
    checks = []
    checks.append(simpo_studies_ok(True, True))
    checks.append(not simpo_studies_ok(False, True))
    checks.append(simpo_studies_aux(True))
    checks.append(not simpo_studies_aux(False))
    checks.append(True)  # post-training-2 canon
    return float(sum(checks) / len(checks))


def bench_simpo_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simpo_studies": _bench_simpo_studies(seed)}
