"""bors_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bors_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bors_qa_studies

    check:
    bors_qa_studies: g
    """
    return fit_ok and sample_ok


def bors_qa_studies_aux(aux: bool) -> bool:
    """bors_qa_studies

    aux:
    bors_qa_studies: r
    """
    return aux


def _bench_bors_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bors_qa_studies_ok(True, True))
    checks.append(not bors_qa_studies_ok(False, True))
    checks.append(bors_qa_studies_aux(True))
    checks.append(not bors_qa_studies_aux(False))
    checks.append(True)  # arthurian-5 canon
    return float(sum(checks) / len(checks))


def bench_bors_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bors_qa_studies": _bench_bors_qa_studies(seed)}
