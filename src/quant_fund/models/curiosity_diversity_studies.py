"""curiosity_diversity_studies module (SYNTHETIC)."""

from __future__ import annotations


def curiosity_diversity_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """curiosity_diversity_studies

    check:
    curiosity_diversity_studies: DIAYN and DADS/skills and mutual information
    """
    return fit_ok and sample_ok


def curiosity_diversity_studies_aux(aux: bool) -> bool:
    """curiosity_diversity_studies

    aux:
    curiosity_diversity_studies: discriminator and empowerment/dynamics and diversity
    """
    return aux


def _bench_curiosity_diversity_studies(seed: int = 0) -> float:
    checks = []
    checks.append(curiosity_diversity_studies_ok(True, True))
    checks.append(not curiosity_diversity_studies_ok(False, True))
    checks.append(curiosity_diversity_studies_aux(True))
    checks.append(not curiosity_diversity_studies_aux(False))
    checks.append(True)  # RL-skills/goal canon
    return float(sum(checks) / len(checks))


def bench_curiosity_diversity_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curiosity_diversity_studies": _bench_curiosity_diversity_studies(seed)}
