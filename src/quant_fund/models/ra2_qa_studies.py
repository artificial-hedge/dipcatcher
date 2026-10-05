"""ra2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ra2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ra2_qa_studies

    check:
    ra2_qa_studies: Ra2QA metrics
    """
    return fit_ok and sample_ok


def ra2_qa_studies_aux(aux: bool) -> bool:
    """ra2_qa_studies

    aux:
    ra2_qa_studies: ra2, sun barks, answers, and scores
    """
    return aux


def _bench_ra2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ra2_qa_studies_ok(True, True))
    checks.append(not ra2_qa_studies_ok(False, True))
    checks.append(ra2_qa_studies_aux(True))
    checks.append(not ra2_qa_studies_aux(False))
    checks.append(True)  # egyptian-7 canon
    return float(sum(checks) / len(checks))


def bench_ra2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ra2_qa_studies": _bench_ra2_qa_studies(seed)}
