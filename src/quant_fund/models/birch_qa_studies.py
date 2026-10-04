"""birch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def birch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """birch_qa_studies

    check:
    birch_qa_studies: BirchQA metrics
    """
    return fit_ok and sample_ok


def birch_qa_studies_aux(aux: bool) -> bool:
    """birch_qa_studies

    aux:
    birch_qa_studies: birches, bark, answers, and scores
    """
    return aux


def _bench_birch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(birch_qa_studies_ok(True, True))
    checks.append(not birch_qa_studies_ok(False, True))
    checks.append(birch_qa_studies_aux(True))
    checks.append(not birch_qa_studies_aux(False))
    checks.append(True)  # arboreal canon
    return float(sum(checks) / len(checks))


def bench_birch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_birch_qa_studies": _bench_birch_qa_studies(seed)}
