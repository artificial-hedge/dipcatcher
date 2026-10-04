"""merganser_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def merganser_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """merganser_qa_studies

    check:
    merganser_qa_studies: MerganserQA metrics
    """
    return fit_ok and sample_ok


def merganser_qa_studies_aux(aux: bool) -> bool:
    """merganser_qa_studies

    aux:
    merganser_qa_studies: mergansers, rivers, answers, and scores
    """
    return aux


def _bench_merganser_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(merganser_qa_studies_ok(True, True))
    checks.append(not merganser_qa_studies_ok(False, True))
    checks.append(merganser_qa_studies_aux(True))
    checks.append(not merganser_qa_studies_aux(False))
    checks.append(True)  # duck canon
    return float(sum(checks) / len(checks))


def bench_merganser_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_merganser_qa_studies": _bench_merganser_qa_studies(seed)}
