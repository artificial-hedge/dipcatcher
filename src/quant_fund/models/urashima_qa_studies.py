"""urashima_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def urashima_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urashima_qa_studies

    check:
    urashima_qa_studies: UrashimaQA metrics
    """
    return fit_ok and sample_ok


def urashima_qa_studies_aux(aux: bool) -> bool:
    """urashima_qa_studies

    aux:
    urashima_qa_studies: urashima, turtle journeys, answers, and scores
    """
    return aux


def _bench_urashima_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(urashima_qa_studies_ok(True, True))
    checks.append(not urashima_qa_studies_ok(False, True))
    checks.append(urashima_qa_studies_aux(True))
    checks.append(not urashima_qa_studies_aux(False))
    checks.append(True)  # japanese-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_urashima_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urashima_qa_studies": _bench_urashima_qa_studies(seed)}
