"""ghost_mantis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ghost_mantis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ghost_mantis_qa_studies

    check:
    ghost_mantis_qa_studies: GhostMantisQA metrics
    """
    return fit_ok and sample_ok


def ghost_mantis_qa_studies_aux(aux: bool) -> bool:
    """ghost_mantis_qa_studies

    aux:
    ghost_mantis_qa_studies: ghost mantises, dead_leaves, answers, and scores
    """
    return aux


def _bench_ghost_mantis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ghost_mantis_qa_studies_ok(True, True))
    checks.append(not ghost_mantis_qa_studies_ok(False, True))
    checks.append(ghost_mantis_qa_studies_aux(True))
    checks.append(not ghost_mantis_qa_studies_aux(False))
    checks.append(True)  # mantis canon
    return float(sum(checks) / len(checks))


def bench_ghost_mantis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ghost_mantis_qa_studies": _bench_ghost_mantis_qa_studies(seed)}
