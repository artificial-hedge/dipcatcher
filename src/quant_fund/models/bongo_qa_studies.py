"""bongo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bongo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bongo_qa_studies

    check:
    bongo_qa_studies: BongoQA metrics
    """
    return fit_ok and sample_ok


def bongo_qa_studies_aux(aux: bool) -> bool:
    """bongo_qa_studies

    aux:
    bongo_qa_studies: bongos, forests, answers, and scores
    """
    return aux


def _bench_bongo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bongo_qa_studies_ok(True, True))
    checks.append(not bongo_qa_studies_ok(False, True))
    checks.append(bongo_qa_studies_aux(True))
    checks.append(not bongo_qa_studies_aux(False))
    checks.append(True)  # antelope-2 canon
    return float(sum(checks) / len(checks))


def bench_bongo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bongo_qa_studies": _bench_bongo_qa_studies(seed)}
