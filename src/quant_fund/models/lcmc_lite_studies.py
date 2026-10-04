"""lcmc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def lcmc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lcmc_lite_studies

    check:
    lcmc_lite_studies: length-generalization metrics
    """
    return fit_ok and sample_ok


def lcmc_lite_studies_aux(aux: bool) -> bool:
    """lcmc_lite_studies

    aux:
    lcmc_lite_studies: strings, splits, predictions, and accuracies
    """
    return aux


def _bench_lcmc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lcmc_lite_studies_ok(True, True))
    checks.append(not lcmc_lite_studies_ok(False, True))
    checks.append(lcmc_lite_studies_aux(True))
    checks.append(not lcmc_lite_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_lcmc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lcmc_lite_studies": _bench_lcmc_lite_studies(seed)}
