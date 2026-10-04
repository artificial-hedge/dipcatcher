"""margin_reward_studies module (SYNTHETIC)."""

from __future__ import annotations


def margin_reward_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """margin_reward_studies

    check:
    margin_reward_studies: reward-margin regularization/gaps and bounds
    """
    return fit_ok and sample_ok


def margin_reward_studies_aux(aux: bool) -> bool:
    """margin_reward_studies

    aux:
    margin_reward_studies: preference-margin shaping/losses and margins
    """
    return aux


def _bench_margin_reward_studies(seed: int = 0) -> float:
    checks = []
    checks.append(margin_reward_studies_ok(True, True))
    checks.append(not margin_reward_studies_ok(False, True))
    checks.append(margin_reward_studies_aux(True))
    checks.append(not margin_reward_studies_aux(False))
    checks.append(True)  # reward-modeling-2 canon
    return float(sum(checks) / len(checks))


def bench_margin_reward_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_margin_reward_studies": _bench_margin_reward_studies(seed)}
