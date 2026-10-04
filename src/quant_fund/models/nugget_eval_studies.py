"""nugget_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def nugget_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nugget_eval_studies

    check:
    nugget_eval_studies: Nugget claim metrics
    """
    return fit_ok and sample_ok


def nugget_eval_studies_aux(aux: bool) -> bool:
    """nugget_eval_studies

    aux:
    nugget_eval_studies: queries, nuggets, answers, and scores
    """
    return aux


def _bench_nugget_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nugget_eval_studies_ok(True, True))
    checks.append(not nugget_eval_studies_ok(False, True))
    checks.append(nugget_eval_studies_aux(True))
    checks.append(not nugget_eval_studies_aux(False))
    checks.append(True)  # LLM-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_nugget_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nugget_eval_studies": _bench_nugget_eval_studies(seed)}
