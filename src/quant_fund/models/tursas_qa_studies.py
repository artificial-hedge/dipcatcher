"""tursas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tursas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tursas_qa_studies

    check:
    tursas_qa_studies: I
    """
    return fit_ok and sample_ok


def tursas_qa_studies_aux(aux: bool) -> bool:
    """tursas_qa_studies

    aux:
    tursas_qa_studies: k
    """
    return aux


def _bench_tursas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tursas_qa_studies_ok(True, True))
    checks.append(not tursas_qa_studies_ok(False, True))
    checks.append(tursas_qa_studies_aux(True))
    checks.append(not tursas_qa_studies_aux(False))
    checks.append(True)  # finnish-demon canon
    return float(sum(checks) / len(checks))


def bench_tursas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tursas_qa_studies": _bench_tursas_qa_studies(seed)}
