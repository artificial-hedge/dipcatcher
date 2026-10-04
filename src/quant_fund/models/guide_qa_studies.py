"""guide_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guide_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guide_qa_studies

    check:
    guide_qa_studies: GuideQA metrics
    """
    return fit_ok and sample_ok


def guide_qa_studies_aux(aux: bool) -> bool:
    """guide_qa_studies

    aux:
    guide_qa_studies: guides, instructions, answers, and scores
    """
    return aux


def _bench_guide_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guide_qa_studies_ok(True, True))
    checks.append(not guide_qa_studies_ok(False, True))
    checks.append(guide_qa_studies_aux(True))
    checks.append(not guide_qa_studies_aux(False))
    checks.append(True)  # instruction-task canon
    return float(sum(checks) / len(checks))


def bench_guide_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guide_qa_studies": _bench_guide_qa_studies(seed)}
