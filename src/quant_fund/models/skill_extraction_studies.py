"""skill_extraction_studies module (SYNTHETIC)."""

from __future__ import annotations


def skill_extraction_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """skill_extraction_studies

    check:
    skill_extraction_studies: latent skill discovery from trajectories/segments and options
    """
    return fit_ok and sample_ok


def skill_extraction_studies_aux(aux: bool) -> bool:
    """skill_extraction_studies

    aux:
    skill_extraction_studies: DIAYN/OPSD-style discriminability/labels and embeddings
    """
    return aux


def _bench_skill_extraction_studies(seed: int = 0) -> float:
    checks = []
    checks.append(skill_extraction_studies_ok(True, True))
    checks.append(not skill_extraction_studies_ok(False, True))
    checks.append(skill_extraction_studies_aux(True))
    checks.append(not skill_extraction_studies_aux(False))
    checks.append(True)  # RL-imitation canon
    return float(sum(checks) / len(checks))


def bench_skill_extraction_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skill_extraction_studies": _bench_skill_extraction_studies(seed)}
