"""peach_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def peach_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """peach_qa_studies

    check:
    peach_qa_studies: PeachQA metrics
    """
    return fit_ok and sample_ok


def peach_qa_studies_aux(aux: bool) -> bool:
    """peach_qa_studies

    aux:
    peach_qa_studies: peaches, fuzz, answers, and scores
    """
    return aux


def _bench_peach_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(peach_qa_studies_ok(True, True))
    checks.append(not peach_qa_studies_ok(False, True))
    checks.append(peach_qa_studies_aux(True))
    checks.append(not peach_qa_studies_aux(False))
    checks.append(True)  # fruit canon
    return float(sum(checks) / len(checks))


def bench_peach_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_peach_qa_studies": _bench_peach_qa_studies(seed)}
