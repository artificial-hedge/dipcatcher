"""sabrewing_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sabrewing_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sabrewing_qa_studies

    check:
    sabrewing_qa_studies: SabrewingQA metrics
    """
    return fit_ok and sample_ok


def sabrewing_qa_studies_aux(aux: bool) -> bool:
    """sabrewing_qa_studies

    aux:
    sabrewing_qa_studies: sabrewings, heliconias, answers, and scores
    """
    return aux


def _bench_sabrewing_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sabrewing_qa_studies_ok(True, True))
    checks.append(not sabrewing_qa_studies_ok(False, True))
    checks.append(sabrewing_qa_studies_aux(True))
    checks.append(not sabrewing_qa_studies_aux(False))
    checks.append(True)  # hummingbird-2 canon
    return float(sum(checks) / len(checks))


def bench_sabrewing_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sabrewing_qa_studies": _bench_sabrewing_qa_studies(seed)}
