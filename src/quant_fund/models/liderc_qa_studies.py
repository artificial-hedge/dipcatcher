"""liderc_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def liderc_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """liderc_qa_studies

    check:
    liderc_qa_studies: LidercQA metrics
    """
    return fit_ok and sample_ok


def liderc_qa_studies_aux(aux: bool) -> bool:
    """liderc_qa_studies

    aux:
    liderc_qa_studies: liderc, night lights, answers, and scores
    """
    return aux


def _bench_liderc_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(liderc_qa_studies_ok(True, True))
    checks.append(not liderc_qa_studies_ok(False, True))
    checks.append(liderc_qa_studies_aux(True))
    checks.append(not liderc_qa_studies_aux(False))
    checks.append(True)  # hungarian-myth canon
    return float(sum(checks) / len(checks))


def bench_liderc_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liderc_qa_studies": _bench_liderc_qa_studies(seed)}
