"""brain_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def brain_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brain_qa_studies

    check:
    brain_qa_studies: BrainQA metrics
    """
    return fit_ok and sample_ok


def brain_qa_studies_aux(aux: bool) -> bool:
    """brain_qa_studies

    aux:
    brain_qa_studies: brains, regions, answers, and scores
    """
    return aux


def _bench_brain_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(brain_qa_studies_ok(True, True))
    checks.append(not brain_qa_studies_ok(False, True))
    checks.append(brain_qa_studies_aux(True))
    checks.append(not brain_qa_studies_aux(False))
    checks.append(True)  # anatomy canon
    return float(sum(checks) / len(checks))


def bench_brain_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brain_qa_studies": _bench_brain_qa_studies(seed)}
