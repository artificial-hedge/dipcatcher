"""illapa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def illapa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """illapa_qa_studies

    check:
    illapa_qa_studies: IllapaQA metrics
    """
    return fit_ok and sample_ok


def illapa_qa_studies_aux(aux: bool) -> bool:
    """illapa_qa_studies

    aux:
    illapa_qa_studies: illapa, thunder gods, answers, and scores
    """
    return aux


def _bench_illapa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(illapa_qa_studies_ok(True, True))
    checks.append(not illapa_qa_studies_ok(False, True))
    checks.append(illapa_qa_studies_aux(True))
    checks.append(not illapa_qa_studies_aux(False))
    checks.append(True)  # incan-myth canon
    return float(sum(checks) / len(checks))


def bench_illapa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_illapa_qa_studies": _bench_illapa_qa_studies(seed)}
