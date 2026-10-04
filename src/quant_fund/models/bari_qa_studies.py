"""bari_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bari_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bari_qa_studies

    check:
    bari_qa_studies: BariQA metrics
    """
    return fit_ok and sample_ok


def bari_qa_studies_aux(aux: bool) -> bool:
    """bari_qa_studies

    aux:
    bari_qa_studies: bari, underworld brides, answers, and scores
    """
    return aux


def _bench_bari_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bari_qa_studies_ok(True, True))
    checks.append(not bari_qa_studies_ok(False, True))
    checks.append(bari_qa_studies_aux(True))
    checks.append(not bari_qa_studies_aux(False))
    checks.append(True)  # korean-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_bari_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bari_qa_studies": _bench_bari_qa_studies(seed)}
