"""eval_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def eval_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """eval_bias_studies

    check:
    eval_bias_studies: Evaluator position/verbosity bias metrics
    """
    return fit_ok and sample_ok


def eval_bias_studies_aux(aux: bool) -> bool:
    """eval_bias_studies

    aux:
    eval_bias_studies: judgments, orderings, and bias scores
    """
    return aux


def _bench_eval_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(eval_bias_studies_ok(True, True))
    checks.append(not eval_bias_studies_ok(False, True))
    checks.append(eval_bias_studies_aux(True))
    checks.append(not eval_bias_studies_aux(False))
    checks.append(True)  # eval-tooling canon
    return float(sum(checks) / len(checks))


def bench_eval_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eval_bias_studies": _bench_eval_bias_studies(seed)}
