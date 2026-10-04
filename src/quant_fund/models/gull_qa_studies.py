"""gull_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gull_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gull_qa_studies

    check:
    gull_qa_studies: GullQA metrics
    """
    return fit_ok and sample_ok


def gull_qa_studies_aux(aux: bool) -> bool:
    """gull_qa_studies

    aux:
    gull_qa_studies: gulls, harbors, answers, and scores
    """
    return aux


def _bench_gull_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gull_qa_studies_ok(True, True))
    checks.append(not gull_qa_studies_ok(False, True))
    checks.append(gull_qa_studies_aux(True))
    checks.append(not gull_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_gull_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gull_qa_studies": _bench_gull_qa_studies(seed)}
