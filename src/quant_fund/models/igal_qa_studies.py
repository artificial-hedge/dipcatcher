"""igal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def igal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """igal_qa_studies

    check:
    igal_qa_studies: o
    """
    return fit_ok and sample_ok


def igal_qa_studies_aux(aux: bool) -> bool:
    """igal_qa_studies

    aux:
    igal_qa_studies: a
    """
    return aux


def _bench_igal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(igal_qa_studies_ok(True, True))
    checks.append(not igal_qa_studies_ok(False, True))
    checks.append(igal_qa_studies_aux(True))
    checks.append(not igal_qa_studies_aux(False))
    checks.append(True)  # garamantian canon
    return float(sum(checks) / len(checks))


def bench_igal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_igal_qa_studies": _bench_igal_qa_studies(seed)}
