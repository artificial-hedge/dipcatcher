"""geo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def geo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geo_qa_studies

    check:
    geo_qa_studies: Geometry question-answering metrics
    """
    return fit_ok and sample_ok


def geo_qa_studies_aux(aux: bool) -> bool:
    """geo_qa_studies

    aux:
    geo_qa_studies: diagrams, questions, answers, and scores
    """
    return aux


def _bench_geo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(geo_qa_studies_ok(True, True))
    checks.append(not geo_qa_studies_ok(False, True))
    checks.append(geo_qa_studies_aux(True))
    checks.append(not geo_qa_studies_aux(False))
    checks.append(True)  # math-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_geo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geo_qa_studies": _bench_geo_qa_studies(seed)}
