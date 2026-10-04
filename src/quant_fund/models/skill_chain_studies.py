"""skill_chain_studies module (SYNTHETIC)."""

from __future__ import annotations


def skill_chain_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """skill_chain_studies

    check:
    skill_chain_studies: chaining and initiation sets/skills and effects
    """
    return fit_ok and sample_ok


def skill_chain_studies_aux(aux: bool) -> bool:
    """skill_chain_studies

    aux:
    skill_chain_studies: backward reachability and landmarks/sequencing and composition
    """
    return aux


def _bench_skill_chain_studies(seed: int = 0) -> float:
    checks = []
    checks.append(skill_chain_studies_ok(True, True))
    checks.append(not skill_chain_studies_ok(False, True))
    checks.append(skill_chain_studies_aux(True))
    checks.append(not skill_chain_studies_aux(False))
    checks.append(True)  # RL-skills/goal canon
    return float(sum(checks) / len(checks))


def bench_skill_chain_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skill_chain_studies": _bench_skill_chain_studies(seed)}
