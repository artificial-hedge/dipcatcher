"""ea_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ea_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ea_qa_studies

    check:
    ea_qa_studies: EaQA metrics
    """
    return fit_ok and sample_ok


def ea_qa_studies_aux(aux: bool) -> bool:
    """ea_qa_studies

    aux:
    ea_qa_studies: ea, deep waters, answers, and scores
    """
    return aux


def _bench_ea_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ea_qa_studies_ok(True, True))
    checks.append(not ea_qa_studies_ok(False, True))
    checks.append(ea_qa_studies_aux(True))
    checks.append(not ea_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-2 canon
    return float(sum(checks) / len(checks))


def bench_ea_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ea_qa_studies": _bench_ea_qa_studies(seed)}
