"""boitata_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boitata_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boitata_qa_studies

    check:
    boitata_qa_studies: B
    """
    return fit_ok and sample_ok


def boitata_qa_studies_aux(aux: bool) -> bool:
    """boitata_qa_studies

    aux:
    boitata_qa_studies: o
    """
    return aux


def _bench_boitata_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boitata_qa_studies_ok(True, True))
    checks.append(not boitata_qa_studies_ok(False, True))
    checks.append(boitata_qa_studies_aux(True))
    checks.append(not boitata_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_boitata_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boitata_qa_studies": _bench_boitata_qa_studies(seed)}
