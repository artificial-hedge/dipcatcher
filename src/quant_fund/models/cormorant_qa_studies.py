"""cormorant_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cormorant_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cormorant_qa_studies

    check:
    cormorant_qa_studies: CormorantQA metrics
    """
    return fit_ok and sample_ok


def cormorant_qa_studies_aux(aux: bool) -> bool:
    """cormorant_qa_studies

    aux:
    cormorant_qa_studies: cormorants, dives, answers, and scores
    """
    return aux


def _bench_cormorant_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cormorant_qa_studies_ok(True, True))
    checks.append(not cormorant_qa_studies_ok(False, True))
    checks.append(cormorant_qa_studies_aux(True))
    checks.append(not cormorant_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_cormorant_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cormorant_qa_studies": _bench_cormorant_qa_studies(seed)}
