"""crowsp_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def crowsp_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crowsp_lite_studies

    check:
    crowsp_lite_studies: CrowS-Pairs-style metrics
    """
    return fit_ok and sample_ok


def crowsp_lite_studies_aux(aux: bool) -> bool:
    """crowsp_lite_studies

    aux:
    crowsp_lite_studies: pairs, stereotypes, predictions, and scores
    """
    return aux


def _bench_crowsp_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crowsp_lite_studies_ok(True, True))
    checks.append(not crowsp_lite_studies_ok(False, True))
    checks.append(crowsp_lite_studies_aux(True))
    checks.append(not crowsp_lite_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_crowsp_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crowsp_lite_studies": _bench_crowsp_lite_studies(seed)}
