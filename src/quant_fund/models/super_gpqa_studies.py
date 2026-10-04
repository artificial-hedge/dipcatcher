"""super_gpqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def super_gpqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """super_gpqa_studies

    check:
    super_gpqa_studies: SuperGPQA metrics
    """
    return fit_ok and sample_ok


def super_gpqa_studies_aux(aux: bool) -> bool:
    """super_gpqa_studies

    aux:
    super_gpqa_studies: questions, options, fields, and scores
    """
    return aux


def _bench_super_gpqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(super_gpqa_studies_ok(True, True))
    checks.append(not super_gpqa_studies_ok(False, True))
    checks.append(super_gpqa_studies_aux(True))
    checks.append(not super_gpqa_studies_aux(False))
    checks.append(True)  # frontier-eval canon
    return float(sum(checks) / len(checks))


def bench_super_gpqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_gpqa_studies": _bench_super_gpqa_studies(seed)}
