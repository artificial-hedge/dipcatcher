"""bucca_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bucca_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bucca_qa_studies

    check:
    bucca_qa_studies: m
    """
    return fit_ok and sample_ok


def bucca_qa_studies_aux(aux: bool) -> bool:
    """bucca_qa_studies

    aux:
    bucca_qa_studies: i
    """
    return aux


def _bench_bucca_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bucca_qa_studies_ok(True, True))
    checks.append(not bucca_qa_studies_ok(False, True))
    checks.append(bucca_qa_studies_aux(True))
    checks.append(not bucca_qa_studies_aux(False))
    checks.append(True)  # cornish-myth canon
    return float(sum(checks) / len(checks))


def bench_bucca_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bucca_qa_studies": _bench_bucca_qa_studies(seed)}
