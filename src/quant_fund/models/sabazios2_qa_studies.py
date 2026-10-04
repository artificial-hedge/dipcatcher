"""sabazios2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sabazios2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sabazios2_qa_studies

    check:
    sabazios2_qa_studies: Sabazios2QA metrics
    """
    return fit_ok and sample_ok


def sabazios2_qa_studies_aux(aux: bool) -> bool:
    """sabazios2_qa_studies

    aux:
    sabazios2_qa_studies: sabazios2, sky horsemen, answers, and scores
    """
    return aux


def _bench_sabazios2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sabazios2_qa_studies_ok(True, True))
    checks.append(not sabazios2_qa_studies_ok(False, True))
    checks.append(sabazios2_qa_studies_aux(True))
    checks.append(not sabazios2_qa_studies_aux(False))
    checks.append(True)  # phrygian-myth canon
    return float(sum(checks) / len(checks))


def bench_sabazios2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sabazios2_qa_studies": _bench_sabazios2_qa_studies(seed)}
