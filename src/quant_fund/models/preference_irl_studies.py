"""preference_irl_studies module (SYNTHETIC)."""

from __future__ import annotations


def preference_irl_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """preference_irl_studies

    check:
    preference_irl_studies: preference-pair reward learning/comparisons and utilities
    """
    return fit_ok and sample_ok


def preference_irl_studies_aux(aux: bool) -> bool:
    """preference_irl_studies

    aux:
    preference_irl_studies: bradley-terry inverse objectives/ranks and partitions
    """
    return aux


def _bench_preference_irl_studies(seed: int = 0) -> float:
    checks = []
    checks.append(preference_irl_studies_ok(True, True))
    checks.append(not preference_irl_studies_ok(False, True))
    checks.append(preference_irl_studies_aux(True))
    checks.append(not preference_irl_studies_aux(False))
    checks.append(True)  # RL-imitation canon
    return float(sum(checks) / len(checks))


def bench_preference_irl_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_preference_irl_studies": _bench_preference_irl_studies(seed)}
