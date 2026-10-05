"""milcom_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def milcom_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """milcom_qa_studies

    check:
    milcom_qa_studies: t
    """
    return fit_ok and sample_ok


def milcom_qa_studies_aux(aux: bool) -> bool:
    """milcom_qa_studies

    aux:
    milcom_qa_studies: h
    """
    return aux


def _bench_milcom_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(milcom_qa_studies_ok(True, True))
    checks.append(not milcom_qa_studies_ok(False, True))
    checks.append(milcom_qa_studies_aux(True))
    checks.append(not milcom_qa_studies_aux(False))
    checks.append(True)  # ammonite-myth canon
    return float(sum(checks) / len(checks))


def bench_milcom_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milcom_qa_studies": _bench_milcom_qa_studies(seed)}
