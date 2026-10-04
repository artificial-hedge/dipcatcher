"""rain_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rain_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rain_qa_studies

    check:
    rain_qa_studies: RainQA metrics
    """
    return fit_ok and sample_ok


def rain_qa_studies_aux(aux: bool) -> bool:
    """rain_qa_studies

    aux:
    rain_qa_studies: rain, precipitation, answers, and scores
    """
    return aux


def _bench_rain_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rain_qa_studies_ok(True, True))
    checks.append(not rain_qa_studies_ok(False, True))
    checks.append(rain_qa_studies_aux(True))
    checks.append(not rain_qa_studies_aux(False))
    checks.append(True)  # weather canon
    return float(sum(checks) / len(checks))


def bench_rain_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rain_qa_studies": _bench_rain_qa_studies(seed)}
