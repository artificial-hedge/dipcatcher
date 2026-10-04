"""mormo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mormo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mormo_qa_studies

    check:
    mormo_qa_studies: m
    """
    return fit_ok and sample_ok


def mormo_qa_studies_aux(aux: bool) -> bool:
    """mormo_qa_studies

    aux:
    mormo_qa_studies: o
    """
    return aux


def _bench_mormo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mormo_qa_studies_ok(True, True))
    checks.append(not mormo_qa_studies_ok(False, True))
    checks.append(mormo_qa_studies_aux(True))
    checks.append(not mormo_qa_studies_aux(False))
    checks.append(True)  # european-vampire canon
    return float(sum(checks) / len(checks))


def bench_mormo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mormo_qa_studies": _bench_mormo_qa_studies(seed)}
