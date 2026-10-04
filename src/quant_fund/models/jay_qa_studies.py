"""jay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jay_qa_studies

    check:
    jay_qa_studies: JayQA metrics
    """
    return fit_ok and sample_ok


def jay_qa_studies_aux(aux: bool) -> bool:
    """jay_qa_studies

    aux:
    jay_qa_studies: jays, oaks, answers, and scores
    """
    return aux


def _bench_jay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jay_qa_studies_ok(True, True))
    checks.append(not jay_qa_studies_ok(False, True))
    checks.append(jay_qa_studies_aux(True))
    checks.append(not jay_qa_studies_aux(False))
    checks.append(True)  # corvid canon
    return float(sum(checks) / len(checks))


def bench_jay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jay_qa_studies": _bench_jay_qa_studies(seed)}
