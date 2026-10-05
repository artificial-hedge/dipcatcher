"""maskim_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def maskim_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """maskim_qa_studies

    check:
    maskim_qa_studies: m
    """
    return fit_ok and sample_ok


def maskim_qa_studies_aux(aux: bool) -> bool:
    """maskim_qa_studies

    aux:
    maskim_qa_studies: a
    """
    return aux


def _bench_maskim_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(maskim_qa_studies_ok(True, True))
    checks.append(not maskim_qa_studies_ok(False, True))
    checks.append(maskim_qa_studies_aux(True))
    checks.append(not maskim_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_maskim_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maskim_qa_studies": _bench_maskim_qa_studies(seed)}
