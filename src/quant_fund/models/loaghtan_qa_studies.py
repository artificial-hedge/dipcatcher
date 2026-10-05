"""loaghtan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def loaghtan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """loaghtan_qa_studies

    check:
    loaghtan_qa_studies: h
    """
    return fit_ok and sample_ok


def loaghtan_qa_studies_aux(aux: bool) -> bool:
    """loaghtan_qa_studies

    aux:
    loaghtan_qa_studies: o
    """
    return aux


def _bench_loaghtan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(loaghtan_qa_studies_ok(True, True))
    checks.append(not loaghtan_qa_studies_ok(False, True))
    checks.append(loaghtan_qa_studies_aux(True))
    checks.append(not loaghtan_qa_studies_aux(False))
    checks.append(True)  # manx-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_loaghtan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loaghtan_qa_studies": _bench_loaghtan_qa_studies(seed)}
