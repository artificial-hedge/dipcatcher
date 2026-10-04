"""recast_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def recast_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """recast_lite_studies

    check:
    recast_lite_studies: Recast NLI metrics
    """
    return fit_ok and sample_ok


def recast_lite_studies_aux(aux: bool) -> bool:
    """recast_lite_studies

    aux:
    recast_lite_studies: contexts, hypotheses, labels, and scores
    """
    return aux


def _bench_recast_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(recast_lite_studies_ok(True, True))
    checks.append(not recast_lite_studies_ok(False, True))
    checks.append(recast_lite_studies_aux(True))
    checks.append(not recast_lite_studies_aux(False))
    checks.append(True)  # intent-paraphrase canon
    return float(sum(checks) / len(checks))


def bench_recast_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_recast_lite_studies": _bench_recast_lite_studies(seed)}
