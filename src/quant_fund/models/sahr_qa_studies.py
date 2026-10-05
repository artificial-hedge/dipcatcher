"""sahr_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sahr_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sahr_qa_studies

    check:
    sahr_qa_studies: m
    """
    return fit_ok and sample_ok


def sahr_qa_studies_aux(aux: bool) -> bool:
    """sahr_qa_studies

    aux:
    sahr_qa_studies: o
    """
    return aux


def _bench_sahr_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sahr_qa_studies_ok(True, True))
    checks.append(not sahr_qa_studies_ok(False, True))
    checks.append(sahr_qa_studies_aux(True))
    checks.append(not sahr_qa_studies_aux(False))
    checks.append(True)  # aramaean-myth canon
    return float(sum(checks) / len(checks))


def bench_sahr_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sahr_qa_studies": _bench_sahr_qa_studies(seed)}
