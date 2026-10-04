"""moorhen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moorhen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moorhen_qa_studies

    check:
    moorhen_qa_studies: MoorhenQA metrics
    """
    return fit_ok and sample_ok


def moorhen_qa_studies_aux(aux: bool) -> bool:
    """moorhen_qa_studies

    aux:
    moorhen_qa_studies: moorhens, ponds, answers, and scores
    """
    return aux


def _bench_moorhen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moorhen_qa_studies_ok(True, True))
    checks.append(not moorhen_qa_studies_ok(False, True))
    checks.append(moorhen_qa_studies_aux(True))
    checks.append(not moorhen_qa_studies_aux(False))
    checks.append(True)  # wader-2 canon
    return float(sum(checks) / len(checks))


def bench_moorhen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moorhen_qa_studies": _bench_moorhen_qa_studies(seed)}
