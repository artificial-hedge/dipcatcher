"""newt_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def newt_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """newt_qa_studies

    check:
    newt_qa_studies: NewtQA metrics
    """
    return fit_ok and sample_ok


def newt_qa_studies_aux(aux: bool) -> bool:
    """newt_qa_studies

    aux:
    newt_qa_studies: newts, tails, answers, and scores
    """
    return aux


def _bench_newt_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(newt_qa_studies_ok(True, True))
    checks.append(not newt_qa_studies_ok(False, True))
    checks.append(newt_qa_studies_aux(True))
    checks.append(not newt_qa_studies_aux(False))
    checks.append(True)  # amphibian canon
    return float(sum(checks) / len(checks))


def bench_newt_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_newt_qa_studies": _bench_newt_qa_studies(seed)}
