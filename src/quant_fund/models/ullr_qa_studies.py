"""ullr_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ullr_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ullr_qa_studies

    check:
    ullr_qa_studies: UllrQA metrics
    """
    return fit_ok and sample_ok


def ullr_qa_studies_aux(aux: bool) -> bool:
    """ullr_qa_studies

    aux:
    ullr_qa_studies: ullr, bow hunters, answers, and scores
    """
    return aux


def _bench_ullr_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ullr_qa_studies_ok(True, True))
    checks.append(not ullr_qa_studies_ok(False, True))
    checks.append(ullr_qa_studies_aux(True))
    checks.append(not ullr_qa_studies_aux(False))
    checks.append(True)  # norse-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_ullr_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ullr_qa_studies": _bench_ullr_qa_studies(seed)}
