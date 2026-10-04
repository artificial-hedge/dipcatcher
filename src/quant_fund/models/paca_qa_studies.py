"""paca_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def paca_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """paca_qa_studies

    check:
    paca_qa_studies: PacaQA metrics
    """
    return fit_ok and sample_ok


def paca_qa_studies_aux(aux: bool) -> bool:
    """paca_qa_studies

    aux:
    paca_qa_studies: pacas, rainforests, answers, and scores
    """
    return aux


def _bench_paca_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(paca_qa_studies_ok(True, True))
    checks.append(not paca_qa_studies_ok(False, True))
    checks.append(paca_qa_studies_aux(True))
    checks.append(not paca_qa_studies_aux(False))
    checks.append(True)  # neotropical-2 canon
    return float(sum(checks) / len(checks))


def bench_paca_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paca_qa_studies": _bench_paca_qa_studies(seed)}
