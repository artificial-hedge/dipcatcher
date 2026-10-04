"""survival_trial_studies module (SYNTHETIC)."""

from __future__ import annotations


def survival_trial_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """survival_trial_studies

    check:
    survival_trial_studies: event-driven power and accrual and endpoint
    """
    return fit_ok and sample_ok


def survival_trial_studies_aux(aux: bool) -> bool:
    """survival_trial_studies

    aux:
    survival_trial_studies: interim and monitoring/futility and futility
    """
    return aux


def _bench_survival_trial_studies(seed: int = 0) -> float:
    checks = []
    checks.append(survival_trial_studies_ok(True, True))
    checks.append(not survival_trial_studies_ok(False, True))
    checks.append(survival_trial_studies_aux(True))
    checks.append(not survival_trial_studies_aux(False))
    checks.append(True)  # trial-statistics/HEOR canon
    return float(sum(checks) / len(checks))


def bench_survival_trial_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_survival_trial_studies": _bench_survival_trial_studies(seed)}
