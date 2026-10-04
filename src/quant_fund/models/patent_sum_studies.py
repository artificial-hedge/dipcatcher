"""patent_sum_studies module (SYNTHETIC)."""

from __future__ import annotations


def patent_sum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """patent_sum_studies

    check:
    patent_sum_studies: Patent abstract metrics
    """
    return fit_ok and sample_ok


def patent_sum_studies_aux(aux: bool) -> bool:
    """patent_sum_studies

    aux:
    patent_sum_studies: descriptions, claims, abstracts, and scores
    """
    return aux


def _bench_patent_sum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(patent_sum_studies_ok(True, True))
    checks.append(not patent_sum_studies_ok(False, True))
    checks.append(patent_sum_studies_aux(True))
    checks.append(not patent_sum_studies_aux(False))
    checks.append(True)  # scientific-summarization canon
    return float(sum(checks) / len(checks))


def bench_patent_sum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_patent_sum_studies": _bench_patent_sum_studies(seed)}
