"""ullr2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ullr2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ullr2_qa_studies

    check:
    ullr2_qa_studies: Ullr2QA metrics
    """
    return fit_ok and sample_ok


def ullr2_qa_studies_aux(aux: bool) -> bool:
    """ullr2_qa_studies

    aux:
    ullr2_qa_studies: ullr2, ski bows, answers, and scores
    """
    return aux


def _bench_ullr2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ullr2_qa_studies_ok(True, True))
    checks.append(not ullr2_qa_studies_ok(False, True))
    checks.append(ullr2_qa_studies_aux(True))
    checks.append(not ullr2_qa_studies_aux(False))
    checks.append(True)  # norse-myth-13 canon
    return float(sum(checks) / len(checks))


def bench_ullr2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ullr2_qa_studies": _bench_ullr2_qa_studies(seed)}
