"""marigold_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def marigold_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marigold_qa_studies

    check:
    marigold_qa_studies: MarigoldQA metrics
    """
    return fit_ok and sample_ok


def marigold_qa_studies_aux(aux: bool) -> bool:
    """marigold_qa_studies

    aux:
    marigold_qa_studies: marigolds, gardens, answers, and scores
    """
    return aux


def _bench_marigold_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marigold_qa_studies_ok(True, True))
    checks.append(not marigold_qa_studies_ok(False, True))
    checks.append(marigold_qa_studies_aux(True))
    checks.append(not marigold_qa_studies_aux(False))
    checks.append(True)  # bloom canon
    return float(sum(checks) / len(checks))


def bench_marigold_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marigold_qa_studies": _bench_marigold_qa_studies(seed)}
