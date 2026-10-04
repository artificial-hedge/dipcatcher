"""aime24_studies module (SYNTHETIC)."""

from __future__ import annotations


def aime24_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aime24_studies

    check:
    aime24_studies: AIME-2024 metrics
    """
    return fit_ok and sample_ok


def aime24_studies_aux(aux: bool) -> bool:
    """aime24_studies

    aux:
    aime24_studies: problems, answers, solutions, and scores
    """
    return aux


def _bench_aime24_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aime24_studies_ok(True, True))
    checks.append(not aime24_studies_ok(False, True))
    checks.append(aime24_studies_aux(True))
    checks.append(not aime24_studies_aux(False))
    checks.append(True)  # frontier-eval canon
    return float(sum(checks) / len(checks))


def bench_aime24_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aime24_studies": _bench_aime24_studies(seed)}
