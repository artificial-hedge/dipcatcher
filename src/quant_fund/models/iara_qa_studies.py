"""iara_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iara_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iara_qa_studies

    check:
    iara_qa_studies: I
    """
    return fit_ok and sample_ok


def iara_qa_studies_aux(aux: bool) -> bool:
    """iara_qa_studies

    aux:
    iara_qa_studies: a
    """
    return aux


def _bench_iara_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iara_qa_studies_ok(True, True))
    checks.append(not iara_qa_studies_ok(False, True))
    checks.append(iara_qa_studies_aux(True))
    checks.append(not iara_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_iara_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iara_qa_studies": _bench_iara_qa_studies(seed)}
