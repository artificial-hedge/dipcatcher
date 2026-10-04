"""downy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def downy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """downy_qa_studies

    check:
    downy_qa_studies: DownyQA metrics
    """
    return fit_ok and sample_ok


def downy_qa_studies_aux(aux: bool) -> bool:
    """downy_qa_studies

    aux:
    downy_qa_studies: downies, feeders, answers, and scores
    """
    return aux


def _bench_downy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(downy_qa_studies_ok(True, True))
    checks.append(not downy_qa_studies_ok(False, True))
    checks.append(downy_qa_studies_aux(True))
    checks.append(not downy_qa_studies_aux(False))
    checks.append(True)  # woodpecker canon
    return float(sum(checks) / len(checks))


def bench_downy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_downy_qa_studies": _bench_downy_qa_studies(seed)}
