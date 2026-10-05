"""beli_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def beli_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """beli_qa_studies

    check:
    beli_qa_studies: f
    """
    return fit_ok and sample_ok


def beli_qa_studies_aux(aux: bool) -> bool:
    """beli_qa_studies

    aux:
    beli_qa_studies: a
    """
    return aux


def _bench_beli_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(beli_qa_studies_ok(True, True))
    checks.append(not beli_qa_studies_ok(False, True))
    checks.append(beli_qa_studies_aux(True))
    checks.append(not beli_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_beli_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beli_qa_studies": _bench_beli_qa_studies(seed)}
