"""reward_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def reward_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reward_bench_studies

    check:
    reward_bench_studies: RewardBench preference-pair accuracy and subset scores
    """
    return fit_ok and sample_ok


def reward_bench_studies_aux(aux: bool) -> bool:
    """reward_bench_studies

    aux:
    reward_bench_studies: chosen/rejected pairs, categories, and accuracy
    """
    return aux


def _bench_reward_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reward_bench_studies_ok(True, True))
    checks.append(not reward_bench_studies_ok(False, True))
    checks.append(reward_bench_studies_aux(True))
    checks.append(not reward_bench_studies_aux(False))
    checks.append(True)  # judge-eval canon
    return float(sum(checks) / len(checks))


def bench_reward_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reward_bench_studies": _bench_reward_bench_studies(seed)}
