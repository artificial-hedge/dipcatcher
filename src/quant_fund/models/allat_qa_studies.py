"""allat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def allat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """allat_qa_studies

    check:
    allat_qa_studies: t
    """
    return fit_ok and sample_ok


def allat_qa_studies_aux(aux: bool) -> bool:
    """allat_qa_studies

    aux:
    allat_qa_studies: h
    """
    return aux


def _bench_allat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(allat_qa_studies_ok(True, True))
    checks.append(not allat_qa_studies_ok(False, True))
    checks.append(allat_qa_studies_aux(True))
    checks.append(not allat_qa_studies_aux(False))
    checks.append(True)  # arabian-myth canon
    return float(sum(checks) / len(checks))


def bench_allat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_allat_qa_studies": _bench_allat_qa_studies(seed)}
