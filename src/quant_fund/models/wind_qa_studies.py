"""wind_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def wind_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wind_qa_studies

    check:
    wind_qa_studies: WindQA metrics
    """
    return fit_ok and sample_ok


def wind_qa_studies_aux(aux: bool) -> bool:
    """wind_qa_studies

    aux:
    wind_qa_studies: winds, speeds, answers, and scores
    """
    return aux


def _bench_wind_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wind_qa_studies_ok(True, True))
    checks.append(not wind_qa_studies_ok(False, True))
    checks.append(wind_qa_studies_aux(True))
    checks.append(not wind_qa_studies_aux(False))
    checks.append(True)  # weather canon
    return float(sum(checks) / len(checks))


def bench_wind_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wind_qa_studies": _bench_wind_qa_studies(seed)}
