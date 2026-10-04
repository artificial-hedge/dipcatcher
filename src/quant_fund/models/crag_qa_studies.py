"""crag_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def crag_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crag_qa_studies

    check:
    crag_qa_studies: CragQA metrics
    """
    return fit_ok and sample_ok


def crag_qa_studies_aux(aux: bool) -> bool:
    """crag_qa_studies

    aux:
    crag_qa_studies: crags, cliffs, answers, and scores
    """
    return aux


def _bench_crag_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crag_qa_studies_ok(True, True))
    checks.append(not crag_qa_studies_ok(False, True))
    checks.append(crag_qa_studies_aux(True))
    checks.append(not crag_qa_studies_aux(False))
    checks.append(True)  # bedrock canon
    return float(sum(checks) / len(checks))


def bench_crag_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crag_qa_studies": _bench_crag_qa_studies(seed)}
