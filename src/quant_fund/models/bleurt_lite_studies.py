"""bleurt_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def bleurt_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bleurt_lite_studies

    check:
    bleurt_lite_studies: BLEURT learned metrics
    """
    return fit_ok and sample_ok


def bleurt_lite_studies_aux(aux: bool) -> bool:
    """bleurt_lite_studies

    aux:
    bleurt_lite_studies: candidates, references, labels, and scores
    """
    return aux


def _bench_bleurt_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bleurt_lite_studies_ok(True, True))
    checks.append(not bleurt_lite_studies_ok(False, True))
    checks.append(bleurt_lite_studies_aux(True))
    checks.append(not bleurt_lite_studies_aux(False))
    checks.append(True)  # generation-metric canon
    return float(sum(checks) / len(checks))


def bench_bleurt_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bleurt_lite_studies": _bench_bleurt_lite_studies(seed)}
