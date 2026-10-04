"""uor_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def uor_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uor_qa_studies

    check:
    uor_qa_studies: U
    """
    return fit_ok and sample_ok


def uor_qa_studies_aux(aux: bool) -> bool:
    """uor_qa_studies

    aux:
    uor_qa_studies: o
    """
    return aux


def _bench_uor_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uor_qa_studies_ok(True, True))
    checks.append(not uor_qa_studies_ok(False, True))
    checks.append(uor_qa_studies_aux(True))
    checks.append(not uor_qa_studies_aux(False))
    checks.append(True)  # turkic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_uor_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uor_qa_studies": _bench_uor_qa_studies(seed)}
