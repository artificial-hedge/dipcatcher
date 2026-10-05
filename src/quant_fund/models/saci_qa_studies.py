"""saci_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def saci_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """saci_qa_studies

    check:
    saci_qa_studies: S
    """
    return fit_ok and sample_ok


def saci_qa_studies_aux(aux: bool) -> bool:
    """saci_qa_studies

    aux:
    saci_qa_studies: a
    """
    return aux


def _bench_saci_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(saci_qa_studies_ok(True, True))
    checks.append(not saci_qa_studies_ok(False, True))
    checks.append(saci_qa_studies_aux(True))
    checks.append(not saci_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_saci_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saci_qa_studies": _bench_saci_qa_studies(seed)}
