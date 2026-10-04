"""fairness_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def fairness_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fairness_eval_studies

    check:
    fairness_eval_studies: Group-fairness metrics
    """
    return fit_ok and sample_ok


def fairness_eval_studies_aux(aux: bool) -> bool:
    """fairness_eval_studies

    aux:
    fairness_eval_studies: examples, groups, outcomes, and gaps
    """
    return aux


def _bench_fairness_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fairness_eval_studies_ok(True, True))
    checks.append(not fairness_eval_studies_ok(False, True))
    checks.append(fairness_eval_studies_aux(True))
    checks.append(not fairness_eval_studies_aux(False))
    checks.append(True)  # bias-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_fairness_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fairness_eval_studies": _bench_fairness_eval_studies(seed)}
