"""aime_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def aime_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aime_eval_studies

    check:
    aime_eval_studies: AIME competition-math metrics
    """
    return fit_ok and sample_ok


def aime_eval_studies_aux(aux: bool) -> bool:
    """aime_eval_studies

    aux:
    aime_eval_studies: problems, answers, solutions, and scores
    """
    return aux


def _bench_aime_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aime_eval_studies_ok(True, True))
    checks.append(not aime_eval_studies_ok(False, True))
    checks.append(aime_eval_studies_aux(True))
    checks.append(not aime_eval_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_aime_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aime_eval_studies": _bench_aime_eval_studies(seed)}
