"""behavior_cloning_studies module (SYNTHETIC)."""

from __future__ import annotations


def behavior_cloning_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """behavior_cloning_studies

    check:
    behavior_cloning_studies: supervised policy fitting to demos/states and actions
    """
    return fit_ok and sample_ok


def behavior_cloning_studies_aux(aux: bool) -> bool:
    """behavior_cloning_studies

    aux:
    behavior_cloning_studies: loss/rollout-gap analysis and covariate shift/drift and compounding
    """
    return aux


def _bench_behavior_cloning_studies(seed: int = 0) -> float:
    checks = []
    checks.append(behavior_cloning_studies_ok(True, True))
    checks.append(not behavior_cloning_studies_ok(False, True))
    checks.append(behavior_cloning_studies_aux(True))
    checks.append(not behavior_cloning_studies_aux(False))
    checks.append(True)  # RL-imitation canon
    return float(sum(checks) / len(checks))


def bench_behavior_cloning_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_behavior_cloning_studies": _bench_behavior_cloning_studies(seed)}
