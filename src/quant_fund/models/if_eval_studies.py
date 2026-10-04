"""if_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def if_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """if_eval_studies

    check:
    if_eval_studies: IFEval instruction-following metrics
    """
    return fit_ok and sample_ok


def if_eval_studies_aux(aux: bool) -> bool:
    """if_eval_studies

    aux:
    if_eval_studies: prompts, constraints, responses, and scores
    """
    return aux


def _bench_if_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(if_eval_studies_ok(True, True))
    checks.append(not if_eval_studies_ok(False, True))
    checks.append(if_eval_studies_aux(True))
    checks.append(not if_eval_studies_aux(False))
    checks.append(True)  # challenge-benchmark canon
    return float(sum(checks) / len(checks))


def bench_if_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_if_eval_studies": _bench_if_eval_studies(seed)}
