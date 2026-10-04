"""topaz_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def topaz_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """topaz_qa_studies

    check:
    topaz_qa_studies: TopazQA metrics
    """
    return fit_ok and sample_ok


def topaz_qa_studies_aux(aux: bool) -> bool:
    """topaz_qa_studies

    aux:
    topaz_qa_studies: topazes, riversides, answers, and scores
    """
    return aux


def _bench_topaz_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(topaz_qa_studies_ok(True, True))
    checks.append(not topaz_qa_studies_ok(False, True))
    checks.append(topaz_qa_studies_aux(True))
    checks.append(not topaz_qa_studies_aux(False))
    checks.append(True)  # hummingbird canon
    return float(sum(checks) / len(checks))


def bench_topaz_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topaz_qa_studies": _bench_topaz_qa_studies(seed)}
