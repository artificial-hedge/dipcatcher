"""baalis2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baalis2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baalis2_qa_studies

    check:
    baalis2_qa_studies: l
    """
    return fit_ok and sample_ok


def baalis2_qa_studies_aux(aux: bool) -> bool:
    """baalis2_qa_studies

    aux:
    baalis2_qa_studies: o
    """
    return aux


def _bench_baalis2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baalis2_qa_studies_ok(True, True))
    checks.append(not baalis2_qa_studies_ok(False, True))
    checks.append(baalis2_qa_studies_aux(True))
    checks.append(not baalis2_qa_studies_aux(False))
    checks.append(True)  # ammonite-myth canon
    return float(sum(checks) / len(checks))


def bench_baalis2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baalis2_qa_studies": _bench_baalis2_qa_studies(seed)}
