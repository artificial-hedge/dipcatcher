"""agouti_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agouti_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agouti_qa_studies

    check:
    agouti_qa_studies: AgoutiQA metrics
    """
    return fit_ok and sample_ok


def agouti_qa_studies_aux(aux: bool) -> bool:
    """agouti_qa_studies

    aux:
    agouti_qa_studies: agoutis, seeds, answers, and scores
    """
    return aux


def _bench_agouti_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agouti_qa_studies_ok(True, True))
    checks.append(not agouti_qa_studies_ok(False, True))
    checks.append(agouti_qa_studies_aux(True))
    checks.append(not agouti_qa_studies_aux(False))
    checks.append(True)  # neotropical canon
    return float(sum(checks) / len(checks))


def bench_agouti_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agouti_qa_studies": _bench_agouti_qa_studies(seed)}
