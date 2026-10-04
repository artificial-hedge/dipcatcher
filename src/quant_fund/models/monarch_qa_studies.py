"""monarch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def monarch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monarch_qa_studies

    check:
    monarch_qa_studies: MonarchQA metrics
    """
    return fit_ok and sample_ok


def monarch_qa_studies_aux(aux: bool) -> bool:
    """monarch_qa_studies

    aux:
    monarch_qa_studies: monarchs, milkweeds, answers, and scores
    """
    return aux


def _bench_monarch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(monarch_qa_studies_ok(True, True))
    checks.append(not monarch_qa_studies_ok(False, True))
    checks.append(monarch_qa_studies_aux(True))
    checks.append(not monarch_qa_studies_aux(False))
    checks.append(True)  # butterfly canon
    return float(sum(checks) / len(checks))


def bench_monarch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monarch_qa_studies": _bench_monarch_qa_studies(seed)}
