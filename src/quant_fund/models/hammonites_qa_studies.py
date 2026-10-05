"""hammonites_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hammonites_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hammonites_qa_studies

    check:
    hammonites_qa_studies: a
    """
    return fit_ok and sample_ok


def hammonites_qa_studies_aux(aux: bool) -> bool:
    """hammonites_qa_studies

    aux:
    hammonites_qa_studies: m
    """
    return aux


def _bench_hammonites_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hammonites_qa_studies_ok(True, True))
    checks.append(not hammonites_qa_studies_ok(False, True))
    checks.append(hammonites_qa_studies_aux(True))
    checks.append(not hammonites_qa_studies_aux(False))
    checks.append(True)  # garamantian-2 canon
    return float(sum(checks) / len(checks))


def bench_hammonites_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hammonites_qa_studies": _bench_hammonites_qa_studies(seed)}
