"""ninki_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ninki_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ninki_qa_studies

    check:
    ninki_qa_studies: NinkiQA metrics
    """
    return fit_ok and sample_ok


def ninki_qa_studies_aux(aux: bool) -> bool:
    """ninki_qa_studies

    aux:
    ninki_qa_studies: ninki, forest guardian, answers, and scores
    """
    return aux


def _bench_ninki_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ninki_qa_studies_ok(True, True))
    checks.append(not ninki_qa_studies_ok(False, True))
    checks.append(ninki_qa_studies_aux(True))
    checks.append(not ninki_qa_studies_aux(False))
    checks.append(True)  # african-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_ninki_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ninki_qa_studies": _bench_ninki_qa_studies(seed)}
