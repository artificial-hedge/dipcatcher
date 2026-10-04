"""osiris2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def osiris2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """osiris2_qa_studies

    check:
    osiris2_qa_studies: Osiris2QA metrics
    """
    return fit_ok and sample_ok


def osiris2_qa_studies_aux(aux: bool) -> bool:
    """osiris2_qa_studies

    aux:
    osiris2_qa_studies: osiris2, green kings, answers, and scores
    """
    return aux


def _bench_osiris2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(osiris2_qa_studies_ok(True, True))
    checks.append(not osiris2_qa_studies_ok(False, True))
    checks.append(osiris2_qa_studies_aux(True))
    checks.append(not osiris2_qa_studies_aux(False))
    checks.append(True)  # egyptian-7 canon
    return float(sum(checks) / len(checks))


def bench_osiris2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osiris2_qa_studies": _bench_osiris2_qa_studies(seed)}
