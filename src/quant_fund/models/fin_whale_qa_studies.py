"""fin_whale_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fin_whale_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fin_whale_qa_studies

    check:
    fin_whale_qa_studies: FinWhaleQA metrics
    """
    return fit_ok and sample_ok


def fin_whale_qa_studies_aux(aux: bool) -> bool:
    """fin_whale_qa_studies

    aux:
    fin_whale_qa_studies: fin whales, open swells, answers, and scores
    """
    return aux


def _bench_fin_whale_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fin_whale_qa_studies_ok(True, True))
    checks.append(not fin_whale_qa_studies_ok(False, True))
    checks.append(fin_whale_qa_studies_aux(True))
    checks.append(not fin_whale_qa_studies_aux(False))
    checks.append(True)  # cetacean canon
    return float(sum(checks) / len(checks))


def bench_fin_whale_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fin_whale_qa_studies": _bench_fin_whale_qa_studies(seed)}
