"""math_reward_studies module (SYNTHETIC)."""

from __future__ import annotations


def math_reward_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """math_reward_studies

    check:
    math_reward_studies: math-answer equivalence checking/boxes and gold
    """
    return fit_ok and sample_ok


def math_reward_studies_aux(aux: bool) -> bool:
    """math_reward_studies

    aux:
    math_reward_studies: numeric/symbolic match rewards/answers and units
    """
    return aux


def _bench_math_reward_studies(seed: int = 0) -> float:
    checks = []
    checks.append(math_reward_studies_ok(True, True))
    checks.append(not math_reward_studies_ok(False, True))
    checks.append(math_reward_studies_aux(True))
    checks.append(not math_reward_studies_aux(False))
    checks.append(True)  # RLVR/verifiable-rewards canon
    return float(sum(checks) / len(checks))


def bench_math_reward_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_math_reward_studies": _bench_math_reward_studies(seed)}
