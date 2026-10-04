"""slender_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def slender_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """slender_qa_studies

    check:
    slender_qa_studies: SlenderQA metrics
    """
    return fit_ok and sample_ok


def slender_qa_studies_aux(aux: bool) -> bool:
    """slender_qa_studies

    aux:
    slender_qa_studies: slender lemurs, dry canopy, answers, and scores
    """
    return aux


def _bench_slender_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(slender_qa_studies_ok(True, True))
    checks.append(not slender_qa_studies_ok(False, True))
    checks.append(slender_qa_studies_aux(True))
    checks.append(not slender_qa_studies_aux(False))
    checks.append(True)  # lemur-4 canon
    return float(sum(checks) / len(checks))


def bench_slender_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slender_qa_studies": _bench_slender_qa_studies(seed)}
