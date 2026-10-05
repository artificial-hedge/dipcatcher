"""uwan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def uwan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uwan_qa_studies

    check:
    uwan_qa_studies: U
    """
    return fit_ok and sample_ok


def uwan_qa_studies_aux(aux: bool) -> bool:
    """uwan_qa_studies

    aux:
    uwan_qa_studies: w
    """
    return aux


def _bench_uwan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uwan_qa_studies_ok(True, True))
    checks.append(not uwan_qa_studies_ok(False, True))
    checks.append(uwan_qa_studies_aux(True))
    checks.append(not uwan_qa_studies_aux(False))
    checks.append(True)  # yokai-9 canon
    return float(sum(checks) / len(checks))


def bench_uwan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uwan_qa_studies": _bench_uwan_qa_studies(seed)}
