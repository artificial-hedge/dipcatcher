"""math500_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def math500_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math500_lite_studies

    check:
    math500_lite_studies: MATH-500 competition metrics
    """
    return fit_ok and sample_ok


def math500_lite_studies_aux(aux: bool) -> bool:
    """math500_lite_studies

    aux:
    math500_lite_studies: problems, solutions, answers, and scores
    """
    return aux


def _bench_math500_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math500_lite_studies_ok(True, True))
    checks.append(not math500_lite_studies_ok(False, True))
    checks.append(math500_lite_studies_aux(True))
    checks.append(not math500_lite_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_math500_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math500_lite_studies": _bench_math500_lite_studies(seed)}
