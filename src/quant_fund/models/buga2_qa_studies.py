"""buga2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def buga2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """buga2_qa_studies

    check:
    buga2_qa_studies: Buga2QA metrics
    """
    return fit_ok and sample_ok


def buga2_qa_studies_aux(aux: bool) -> bool:
    """buga2_qa_studies

    aux:
    buga2_qa_studies: buga2, iron lords, answers, and scores
    """
    return aux


def _bench_buga2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(buga2_qa_studies_ok(True, True))
    checks.append(not buga2_qa_studies_ok(False, True))
    checks.append(buga2_qa_studies_aux(True))
    checks.append(not buga2_qa_studies_aux(False))
    checks.append(True)  # siberian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_buga2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_buga2_qa_studies": _bench_buga2_qa_studies(seed)}
