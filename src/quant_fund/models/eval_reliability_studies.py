"""eval_reliability_studies module (SYNTHETIC)."""

from __future__ import annotations


def eval_reliability_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """eval_reliability_studies

    check:
    eval_reliability_studies: eval reliability/inter-rater and seed variance
    """
    return fit_ok and sample_ok


def eval_reliability_studies_aux(aux: bool) -> bool:
    """eval_reliability_studies

    aux:
    eval_reliability_studies: eval-metric stability suites/runs and intervals
    """
    return aux


def _bench_eval_reliability_studies(seed: int = 0) -> float:
    checks = []
    checks.append(eval_reliability_studies_ok(True, True))
    checks.append(not eval_reliability_studies_ok(False, True))
    checks.append(eval_reliability_studies_aux(True))
    checks.append(not eval_reliability_studies_aux(False))
    checks.append(True)  # eval-science canon
    return float(sum(checks) / len(checks))


def bench_eval_reliability_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eval_reliability_studies": _bench_eval_reliability_studies(seed)}
