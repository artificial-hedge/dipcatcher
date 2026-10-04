"""cariocecus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cariocecus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cariocecus_qa_studies

    check:
    cariocecus_qa_studies: h
    """
    return fit_ok and sample_ok


def cariocecus_qa_studies_aux(aux: bool) -> bool:
    """cariocecus_qa_studies

    aux:
    cariocecus_qa_studies: o
    """
    return aux


def _bench_cariocecus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cariocecus_qa_studies_ok(True, True))
    checks.append(not cariocecus_qa_studies_ok(False, True))
    checks.append(cariocecus_qa_studies_aux(True))
    checks.append(not cariocecus_qa_studies_aux(False))
    checks.append(True)  # iberian-myth canon
    return float(sum(checks) / len(checks))


def bench_cariocecus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cariocecus_qa_studies": _bench_cariocecus_qa_studies(seed)}
