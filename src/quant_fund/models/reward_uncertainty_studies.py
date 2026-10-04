"""reward_uncertainty_studies module (SYNTHETIC)."""

from __future__ import annotations


def reward_uncertainty_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reward_uncertainty_studies

    check:
    reward_uncertainty_studies: epistemic RM uncertainty bands/heads and drops
    """
    return fit_ok and sample_ok


def reward_uncertainty_studies_aux(aux: bool) -> bool:
    """reward_uncertainty_studies

    aux:
    reward_uncertainty_studies: reward-score confidence calibration/scores and intervals
    """
    return aux


def _bench_reward_uncertainty_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reward_uncertainty_studies_ok(True, True))
    checks.append(not reward_uncertainty_studies_ok(False, True))
    checks.append(reward_uncertainty_studies_aux(True))
    checks.append(not reward_uncertainty_studies_aux(False))
    checks.append(True)  # reward-modeling-2 canon
    return float(sum(checks) / len(checks))


def bench_reward_uncertainty_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reward_uncertainty_studies": _bench_reward_uncertainty_studies(seed)}
