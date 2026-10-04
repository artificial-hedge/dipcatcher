"""outcome_reward_studies module (SYNTHETIC)."""

from __future__ import annotations


def outcome_reward_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """outcome_reward_studies

    check:
    outcome_reward_studies: outcome-level reward signals/correct and finals
    """
    return fit_ok and sample_ok


def outcome_reward_studies_aux(aux: bool) -> bool:
    """outcome_reward_studies

    aux:
    outcome_reward_studies: ORM scoring over full trajectories/endings and votes
    """
    return aux


def _bench_outcome_reward_studies(seed: int = 0) -> float:
    checks = []
    checks.append(outcome_reward_studies_ok(True, True))
    checks.append(not outcome_reward_studies_ok(False, True))
    checks.append(outcome_reward_studies_aux(True))
    checks.append(not outcome_reward_studies_aux(False))
    checks.append(True)  # RLVR/verifiable-rewards canon
    return float(sum(checks) / len(checks))


def bench_outcome_reward_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_outcome_reward_studies": _bench_outcome_reward_studies(seed)}
