"""capability_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def capability_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """capability_eval_studies

    check:
    capability_eval_studies: dangerous-capability assessments/tasks and thresholds
    """
    return fit_ok and sample_ok


def capability_eval_studies_aux(aux: bool) -> bool:
    """capability_eval_studies

    aux:
    capability_eval_studies: eval elicitation and uplift measurement/scores and baselines
    """
    return aux


def _bench_capability_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(capability_eval_studies_ok(True, True))
    checks.append(not capability_eval_studies_ok(False, True))
    checks.append(capability_eval_studies_aux(True))
    checks.append(not capability_eval_studies_aux(False))
    checks.append(True)  # agent-safety canon
    return float(sum(checks) / len(checks))


def bench_capability_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_capability_eval_studies": _bench_capability_eval_studies(seed)}
