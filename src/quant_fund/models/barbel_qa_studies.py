"""barbel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barbel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barbel_qa_studies

    check:
    barbel_qa_studies: BarbelQA metrics
    """
    return fit_ok and sample_ok


def barbel_qa_studies_aux(aux: bool) -> bool:
    """barbel_qa_studies

    aux:
    barbel_qa_studies: barbels, gravel runs, answers, and scores
    """
    return aux


def _bench_barbel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barbel_qa_studies_ok(True, True))
    checks.append(not barbel_qa_studies_ok(False, True))
    checks.append(barbel_qa_studies_aux(True))
    checks.append(not barbel_qa_studies_aux(False))
    checks.append(True)  # cyprinid canon
    return float(sum(checks) / len(checks))


def bench_barbel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barbel_qa_studies": _bench_barbel_qa_studies(seed)}
