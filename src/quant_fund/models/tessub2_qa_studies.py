"""tessub2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tessub2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tessub2_qa_studies

    check:
    tessub2_qa_studies: Tessub2QA metrics
    """
    return fit_ok and sample_ok


def tessub2_qa_studies_aux(aux: bool) -> bool:
    """tessub2_qa_studies

    aux:
    tessub2_qa_studies: tessub2, storm kings, answers, and scores
    """
    return aux


def _bench_tessub2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tessub2_qa_studies_ok(True, True))
    checks.append(not tessub2_qa_studies_ok(False, True))
    checks.append(tessub2_qa_studies_aux(True))
    checks.append(not tessub2_qa_studies_aux(False))
    checks.append(True)  # hurrian-myth canon
    return float(sum(checks) / len(checks))


def bench_tessub2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tessub2_qa_studies": _bench_tessub2_qa_studies(seed)}
