"""step_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def step_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """step_eval_studies

    check:
    step_eval_studies: step-wise verifier scores and per-step accuracy
    """
    return fit_ok and sample_ok


def step_eval_studies_aux(aux: bool) -> bool:
    """step_eval_studies

    aux:
    step_eval_studies: reasoning steps, labels, and step metrics
    """
    return aux


def _bench_step_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(step_eval_studies_ok(True, True))
    checks.append(not step_eval_studies_ok(False, True))
    checks.append(step_eval_studies_aux(True))
    checks.append(not step_eval_studies_aux(False))
    checks.append(True)  # eval-science-2 canon
    return float(sum(checks) / len(checks))


def bench_step_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_step_eval_studies": _bench_step_eval_studies(seed)}
