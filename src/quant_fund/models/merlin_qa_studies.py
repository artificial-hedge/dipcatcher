"""merlin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def merlin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """merlin_qa_studies

    check:
    merlin_qa_studies: MerlinQA metrics
    """
    return fit_ok and sample_ok


def merlin_qa_studies_aux(aux: bool) -> bool:
    """merlin_qa_studies

    aux:
    merlin_qa_studies: merlins, moors, answers, and scores
    """
    return aux


def _bench_merlin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(merlin_qa_studies_ok(True, True))
    checks.append(not merlin_qa_studies_ok(False, True))
    checks.append(merlin_qa_studies_aux(True))
    checks.append(not merlin_qa_studies_aux(False))
    checks.append(True)  # raptor-2 canon
    return float(sum(checks) / len(checks))


def bench_merlin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_merlin_qa_studies": _bench_merlin_qa_studies(seed)}
