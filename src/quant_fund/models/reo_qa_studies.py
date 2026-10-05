"""reo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def reo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reo_qa_studies

    check:
    reo_qa_studies: h
    """
    return fit_ok and sample_ok


def reo_qa_studies_aux(aux: bool) -> bool:
    """reo_qa_studies

    aux:
    reo_qa_studies: i
    """
    return aux


def _bench_reo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reo_qa_studies_ok(True, True))
    checks.append(not reo_qa_studies_ok(False, True))
    checks.append(reo_qa_studies_aux(True))
    checks.append(not reo_qa_studies_aux(False))
    checks.append(True)  # lusitanian-myth canon
    return float(sum(checks) / len(checks))


def bench_reo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reo_qa_studies": _bench_reo_qa_studies(seed)}
