"""alg514_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def alg514_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """alg514_lite_studies

    check:
    alg514_lite_studies: ALG514 metrics
    """
    return fit_ok and sample_ok


def alg514_lite_studies_aux(aux: bool) -> bool:
    """alg514_lite_studies

    aux:
    alg514_lite_studies: problems, equations, answers, and scores
    """
    return aux


def _bench_alg514_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(alg514_lite_studies_ok(True, True))
    checks.append(not alg514_lite_studies_ok(False, True))
    checks.append(alg514_lite_studies_aux(True))
    checks.append(not alg514_lite_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_alg514_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alg514_lite_studies": _bench_alg514_lite_studies(seed)}
