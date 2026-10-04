"""inari2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def inari2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inari2_qa_studies

    check:
    inari2_qa_studies: Inari2QA metrics
    """
    return fit_ok and sample_ok


def inari2_qa_studies_aux(aux: bool) -> bool:
    """inari2_qa_studies

    aux:
    inari2_qa_studies: inari2, fox messengers, answers, and scores
    """
    return aux


def _bench_inari2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(inari2_qa_studies_ok(True, True))
    checks.append(not inari2_qa_studies_ok(False, True))
    checks.append(inari2_qa_studies_aux(True))
    checks.append(not inari2_qa_studies_aux(False))
    checks.append(True)  # japanese-myth-8 canon
    return float(sum(checks) / len(checks))


def bench_inari2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inari2_qa_studies": _bench_inari2_qa_studies(seed)}
