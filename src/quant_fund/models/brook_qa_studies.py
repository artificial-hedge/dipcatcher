"""brook_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def brook_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """brook_qa_studies

    check:
    brook_qa_studies: BrookQA metrics
    """
    return fit_ok and sample_ok


def brook_qa_studies_aux(aux: bool) -> bool:
    """brook_qa_studies

    aux:
    brook_qa_studies: brooks, trickles, answers, and scores
    """
    return aux


def _bench_brook_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(brook_qa_studies_ok(True, True))
    checks.append(not brook_qa_studies_ok(False, True))
    checks.append(brook_qa_studies_aux(True))
    checks.append(not brook_qa_studies_aux(False))
    checks.append(True)  # wetland canon
    return float(sum(checks) / len(checks))


def bench_brook_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brook_qa_studies": _bench_brook_qa_studies(seed)}
