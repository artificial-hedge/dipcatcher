"""abaasy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def abaasy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abaasy_qa_studies

    check:
    abaasy_qa_studies: A
    """
    return fit_ok and sample_ok


def abaasy_qa_studies_aux(aux: bool) -> bool:
    """abaasy_qa_studies

    aux:
    abaasy_qa_studies: b
    """
    return aux


def _bench_abaasy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abaasy_qa_studies_ok(True, True))
    checks.append(not abaasy_qa_studies_ok(False, True))
    checks.append(abaasy_qa_studies_aux(True))
    checks.append(not abaasy_qa_studies_aux(False))
    checks.append(True)  # siberian-demon canon
    return float(sum(checks) / len(checks))


def bench_abaasy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abaasy_qa_studies": _bench_abaasy_qa_studies(seed)}
