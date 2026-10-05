"""ninurta2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ninurta2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ninurta2_qa_studies

    check:
    ninurta2_qa_studies: Ninurta2QA metrics
    """
    return fit_ok and sample_ok


def ninurta2_qa_studies_aux(aux: bool) -> bool:
    """ninurta2_qa_studies

    aux:
    ninurta2_qa_studies: ninurta2, storm hunters, answers, and scores
    """
    return aux


def _bench_ninurta2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ninurta2_qa_studies_ok(True, True))
    checks.append(not ninurta2_qa_studies_ok(False, True))
    checks.append(ninurta2_qa_studies_aux(True))
    checks.append(not ninurta2_qa_studies_aux(False))
    checks.append(True)  # sumerian-6 canon
    return float(sum(checks) / len(checks))


def bench_ninurta2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ninurta2_qa_studies": _bench_ninurta2_qa_studies(seed)}
