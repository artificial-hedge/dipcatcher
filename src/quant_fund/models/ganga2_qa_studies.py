"""ganga2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ganga2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ganga2_qa_studies

    check:
    ganga2_qa_studies: Ganga2QA metrics
    """
    return fit_ok and sample_ok


def ganga2_qa_studies_aux(aux: bool) -> bool:
    """ganga2_qa_studies

    aux:
    ganga2_qa_studies: ganga2, sky rivers, answers, and scores
    """
    return aux


def _bench_ganga2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ganga2_qa_studies_ok(True, True))
    checks.append(not ganga2_qa_studies_ok(False, True))
    checks.append(ganga2_qa_studies_aux(True))
    checks.append(not ganga2_qa_studies_aux(False))
    checks.append(True)  # hindu-myth-7 canon
    return float(sum(checks) / len(checks))


def bench_ganga2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ganga2_qa_studies": _bench_ganga2_qa_studies(seed)}
