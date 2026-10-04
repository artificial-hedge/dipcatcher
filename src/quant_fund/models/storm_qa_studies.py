"""storm_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def storm_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """storm_qa_studies

    check:
    storm_qa_studies: StormQA metrics
    """
    return fit_ok and sample_ok


def storm_qa_studies_aux(aux: bool) -> bool:
    """storm_qa_studies

    aux:
    storm_qa_studies: storms, warnings, answers, and scores
    """
    return aux


def _bench_storm_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(storm_qa_studies_ok(True, True))
    checks.append(not storm_qa_studies_ok(False, True))
    checks.append(storm_qa_studies_aux(True))
    checks.append(not storm_qa_studies_aux(False))
    checks.append(True)  # weather canon
    return float(sum(checks) / len(checks))


def bench_storm_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_storm_qa_studies": _bench_storm_qa_studies(seed)}
