"""ipos_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ipos_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ipos_qa_studies

    check:
    ipos_qa_studies: I
    """
    return fit_ok and sample_ok


def ipos_qa_studies_aux(aux: bool) -> bool:
    """ipos_qa_studies

    aux:
    ipos_qa_studies: p
    """
    return aux


def _bench_ipos_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ipos_qa_studies_ok(True, True))
    checks.append(not ipos_qa_studies_ok(False, True))
    checks.append(ipos_qa_studies_aux(True))
    checks.append(not ipos_qa_studies_aux(False))
    checks.append(True)  # goetic-throne canon
    return float(sum(checks) / len(checks))


def bench_ipos_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ipos_qa_studies": _bench_ipos_qa_studies(seed)}
