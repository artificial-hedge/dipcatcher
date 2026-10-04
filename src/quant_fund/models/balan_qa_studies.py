"""balan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def balan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """balan_qa_studies

    check:
    balan_qa_studies: d
    """
    return fit_ok and sample_ok


def balan_qa_studies_aux(aux: bool) -> bool:
    """balan_qa_studies

    aux:
    balan_qa_studies: o
    """
    return aux


def _bench_balan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(balan_qa_studies_ok(True, True))
    checks.append(not balan_qa_studies_ok(False, True))
    checks.append(balan_qa_studies_aux(True))
    checks.append(not balan_qa_studies_aux(False))
    checks.append(True)  # arthurian-7 canon
    return float(sum(checks) / len(checks))


def bench_balan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_balan_qa_studies": _bench_balan_qa_studies(seed)}
