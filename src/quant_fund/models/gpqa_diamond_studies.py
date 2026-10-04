"""gpqa_diamond_studies module (SYNTHETIC)."""

from __future__ import annotations


def gpqa_diamond_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gpqa_diamond_studies

    check:
    gpqa_diamond_studies: GPQA-diamond metrics
    """
    return fit_ok and sample_ok


def gpqa_diamond_studies_aux(aux: bool) -> bool:
    """gpqa_diamond_studies

    aux:
    gpqa_diamond_studies: questions, options, explanations, and scores
    """
    return aux


def _bench_gpqa_diamond_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gpqa_diamond_studies_ok(True, True))
    checks.append(not gpqa_diamond_studies_ok(False, True))
    checks.append(gpqa_diamond_studies_aux(True))
    checks.append(not gpqa_diamond_studies_aux(False))
    checks.append(True)  # frontier-eval canon
    return float(sum(checks) / len(checks))


def bench_gpqa_diamond_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gpqa_diamond_studies": _bench_gpqa_diamond_studies(seed)}
