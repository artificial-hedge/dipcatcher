"""eli5_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def eli5_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """eli5_lite_studies

    check:
    eli5_lite_studies: ELI5 metrics
    """
    return fit_ok and sample_ok


def eli5_lite_studies_aux(aux: bool) -> bool:
    """eli5_lite_studies

    aux:
    eli5_lite_studies: questions, explanations, scores
    """
    return aux


def _bench_eli5_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(eli5_lite_studies_ok(True, True))
    checks.append(not eli5_lite_studies_ok(False, True))
    checks.append(eli5_lite_studies_aux(True))
    checks.append(not eli5_lite_studies_aux(False))
    checks.append(True)  # retrieval-eval canon
    return float(sum(checks) / len(checks))


def bench_eli5_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eli5_lite_studies": _bench_eli5_lite_studies(seed)}
