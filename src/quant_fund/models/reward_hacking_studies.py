"""reward_hacking_studies module (SYNTHETIC)."""

from __future__ import annotations


def reward_hacking_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reward_hacking_studies

    check:
    reward_hacking_studies: reward-model exploitation detection/scores and spikes
    """
    return fit_ok and sample_ok


def reward_hacking_studies_aux(aux: bool) -> bool:
    """reward_hacking_studies

    aux:
    reward_hacking_studies: goodhart probe suites/inputs and artifacts
    """
    return aux


def _bench_reward_hacking_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reward_hacking_studies_ok(True, True))
    checks.append(not reward_hacking_studies_ok(False, True))
    checks.append(reward_hacking_studies_aux(True))
    checks.append(not reward_hacking_studies_aux(False))
    checks.append(True)  # reward-modeling-2 canon
    return float(sum(checks) / len(checks))


def bench_reward_hacking_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reward_hacking_studies": _bench_reward_hacking_studies(seed)}
