"""diving_petrel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def diving_petrel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diving_petrel_qa_studies

    check:
    diving_petrel_qa_studies: Diving-petrelQA metrics
    """
    return fit_ok and sample_ok


def diving_petrel_qa_studies_aux(aux: bool) -> bool:
    """diving_petrel_qa_studies

    aux:
    diving_petrel_qa_studies: diving petrels, fjords, answers, and scores
    """
    return aux


def _bench_diving_petrel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diving_petrel_qa_studies_ok(True, True))
    checks.append(not diving_petrel_qa_studies_ok(False, True))
    checks.append(diving_petrel_qa_studies_aux(True))
    checks.append(not diving_petrel_qa_studies_aux(False))
    checks.append(True)  # pelagic canon
    return float(sum(checks) / len(checks))


def bench_diving_petrel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diving_petrel_qa_studies": _bench_diving_petrel_qa_studies(seed)}
