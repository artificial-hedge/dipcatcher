"""usnea_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def usnea_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """usnea_qa_studies

    check:
    usnea_qa_studies: UsneaQA metrics
    """
    return fit_ok and sample_ok


def usnea_qa_studies_aux(aux: bool) -> bool:
    """usnea_qa_studies

    aux:
    usnea_qa_studies: usneas, branches, answers, and scores
    """
    return aux


def _bench_usnea_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(usnea_qa_studies_ok(True, True))
    checks.append(not usnea_qa_studies_ok(False, True))
    checks.append(usnea_qa_studies_aux(True))
    checks.append(not usnea_qa_studies_aux(False))
    checks.append(True)  # lichen canon
    return float(sum(checks) / len(checks))


def bench_usnea_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_usnea_qa_studies": _bench_usnea_qa_studies(seed)}
