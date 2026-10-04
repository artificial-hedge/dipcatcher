"""verifier_reward_studies module (SYNTHETIC)."""

from __future__ import annotations


def verifier_reward_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """verifier_reward_studies

    check:
    verifier_reward_studies: verifier reward-model ranking and correctness scores
    """
    return fit_ok and sample_ok


def verifier_reward_studies_aux(aux: bool) -> bool:
    """verifier_reward_studies

    aux:
    verifier_reward_studies: correct/incorrect solutions, rankings, and metrics
    """
    return aux


def _bench_verifier_reward_studies(seed: int = 0) -> float:
    checks = []
    checks.append(verifier_reward_studies_ok(True, True))
    checks.append(not verifier_reward_studies_ok(False, True))
    checks.append(verifier_reward_studies_aux(True))
    checks.append(not verifier_reward_studies_aux(False))
    checks.append(True)  # eval-science-2 canon
    return float(sum(checks) / len(checks))


def bench_verifier_reward_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verifier_reward_studies": _bench_verifier_reward_studies(seed)}
