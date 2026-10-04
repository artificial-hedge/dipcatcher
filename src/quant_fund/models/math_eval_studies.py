"""math_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_eval_studies

    check:
    math_eval_studies: MathEval metrics
    """
    return fit_ok and sample_ok


def math_eval_studies_aux(aux: bool) -> bool:
    """math_eval_studies

    aux:
    math_eval_studies: problems, solutions, answers, and scores
    """
    return aux


def _bench_math_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_eval_studies_ok(True, True))
    checks.append(not math_eval_studies_ok(False, True))
    checks.append(math_eval_studies_aux(True))
    checks.append(not math_eval_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_math_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_eval_studies": _bench_math_eval_studies(seed)}
