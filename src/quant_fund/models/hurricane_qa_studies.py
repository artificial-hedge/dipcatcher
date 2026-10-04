"""hurricane_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hurricane_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hurricane_qa_studies

    check:
    hurricane_qa_studies: HurricaneQA metrics
    """
    return fit_ok and sample_ok


def hurricane_qa_studies_aux(aux: bool) -> bool:
    """hurricane_qa_studies

    aux:
    hurricane_qa_studies: hurricanes, intensities, answers, and scores
    """
    return aux


def _bench_hurricane_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hurricane_qa_studies_ok(True, True))
    checks.append(not hurricane_qa_studies_ok(False, True))
    checks.append(hurricane_qa_studies_aux(True))
    checks.append(not hurricane_qa_studies_aux(False))
    checks.append(True)  # weather canon
    return float(sum(checks) / len(checks))


def bench_hurricane_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hurricane_qa_studies": _bench_hurricane_qa_studies(seed)}
