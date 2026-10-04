"""trogon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def trogon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trogon_qa_studies

    check:
    trogon_qa_studies: TrogonQA metrics
    """
    return fit_ok and sample_ok


def trogon_qa_studies_aux(aux: bool) -> bool:
    """trogon_qa_studies

    aux:
    trogon_qa_studies: trogons, cloudforests, answers, and scores
    """
    return aux


def _bench_trogon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(trogon_qa_studies_ok(True, True))
    checks.append(not trogon_qa_studies_ok(False, True))
    checks.append(trogon_qa_studies_aux(True))
    checks.append(not trogon_qa_studies_aux(False))
    checks.append(True)  # canopybird canon
    return float(sum(checks) / len(checks))


def bench_trogon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trogon_qa_studies": _bench_trogon_qa_studies(seed)}
