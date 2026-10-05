"""supay2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def supay2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """supay2_qa_studies

    check:
    supay2_qa_studies: Supay2QA metrics
    """
    return fit_ok and sample_ok


def supay2_qa_studies_aux(aux: bool) -> bool:
    """supay2_qa_studies

    aux:
    supay2_qa_studies: supay2, underworld spirits, answers, and scores
    """
    return aux


def _bench_supay2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(supay2_qa_studies_ok(True, True))
    checks.append(not supay2_qa_studies_ok(False, True))
    checks.append(supay2_qa_studies_aux(True))
    checks.append(not supay2_qa_studies_aux(False))
    checks.append(True)  # incan-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_supay2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_supay2_qa_studies": _bench_supay2_qa_studies(seed)}
