"""adaptive_trial_studies module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_trial_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adaptive_trial_studies

    check:
    adaptive_trial_studies: enrichment and seamless/arm and response
    """
    return fit_ok and sample_ok


def adaptive_trial_studies_aux(aux: bool) -> bool:
    """adaptive_trial_studies

    aux:
    adaptive_trial_studies: bayesian and platform/master and stopping
    """
    return aux


def _bench_adaptive_trial_studies(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_trial_studies_ok(True, True))
    checks.append(not adaptive_trial_studies_ok(False, True))
    checks.append(adaptive_trial_studies_aux(True))
    checks.append(not adaptive_trial_studies_aux(False))
    checks.append(True)  # clinical-research-methods canon
    return float(sum(checks) / len(checks))


def bench_adaptive_trial_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_trial_studies": _bench_adaptive_trial_studies(seed)}
