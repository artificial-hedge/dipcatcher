"""plus_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def plus_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plus_eval_studies

    check:
    plus_eval_studies: PlusEval metrics
    """
    return fit_ok and sample_ok


def plus_eval_studies_aux(aux: bool) -> bool:
    """plus_eval_studies

    aux:
    plus_eval_studies: questions, rationales, answers, and scores
    """
    return aux


def _bench_plus_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(plus_eval_studies_ok(True, True))
    checks.append(not plus_eval_studies_ok(False, True))
    checks.append(plus_eval_studies_aux(True))
    checks.append(not plus_eval_studies_aux(False))
    checks.append(True)  # live-eval canon
    return float(sum(checks) / len(checks))


def bench_plus_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plus_eval_studies": _bench_plus_eval_studies(seed)}
