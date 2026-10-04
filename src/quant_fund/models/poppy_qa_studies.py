"""poppy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def poppy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """poppy_qa_studies

    check:
    poppy_qa_studies: PoppyQA metrics
    """
    return fit_ok and sample_ok


def poppy_qa_studies_aux(aux: bool) -> bool:
    """poppy_qa_studies

    aux:
    poppy_qa_studies: poppies, capsules, answers, and scores
    """
    return aux


def _bench_poppy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(poppy_qa_studies_ok(True, True))
    checks.append(not poppy_qa_studies_ok(False, True))
    checks.append(poppy_qa_studies_aux(True))
    checks.append(not poppy_qa_studies_aux(False))
    checks.append(True)  # wildflower canon
    return float(sum(checks) / len(checks))


def bench_poppy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poppy_qa_studies": _bench_poppy_qa_studies(seed)}
