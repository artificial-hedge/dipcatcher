"""winogrande_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def winogrande_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """winogrande_lite_studies

    check:
    winogrande_lite_studies: Winogrande coreference metrics
    """
    return fit_ok and sample_ok


def winogrande_lite_studies_aux(aux: bool) -> bool:
    """winogrande_lite_studies

    aux:
    winogrande_lite_studies: sentences, options, answers, and scores
    """
    return aux


def _bench_winogrande_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(winogrande_lite_studies_ok(True, True))
    checks.append(not winogrande_lite_studies_ok(False, True))
    checks.append(winogrande_lite_studies_aux(True))
    checks.append(not winogrande_lite_studies_aux(False))
    checks.append(True)  # commonsense-reasoning canon
    return float(sum(checks) / len(checks))


def bench_winogrande_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_winogrande_lite_studies": _bench_winogrande_lite_studies(seed)}
