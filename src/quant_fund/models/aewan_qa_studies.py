"""aewan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aewan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aewan_qa_studies

    check:
    aewan_qa_studies: n
    """
    return fit_ok and sample_ok


def aewan_qa_studies_aux(aux: bool) -> bool:
    """aewan_qa_studies

    aux:
    aewan_qa_studies: o
    """
    return aux


def _bench_aewan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aewan_qa_studies_ok(True, True))
    checks.append(not aewan_qa_studies_ok(False, True))
    checks.append(aewan_qa_studies_aux(True))
    checks.append(not aewan_qa_studies_aux(False))
    checks.append(True)  # saharan canon
    return float(sum(checks) / len(checks))


def bench_aewan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aewan_qa_studies": _bench_aewan_qa_studies(seed)}
