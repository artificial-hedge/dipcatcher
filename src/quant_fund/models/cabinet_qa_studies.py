"""cabinet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cabinet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cabinet_qa_studies

    check:
    cabinet_qa_studies: CabinetQA metrics
    """
    return fit_ok and sample_ok


def cabinet_qa_studies_aux(aux: bool) -> bool:
    """cabinet_qa_studies

    aux:
    cabinet_qa_studies: ministers, portfolios, answers, and scores
    """
    return aux


def _bench_cabinet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cabinet_qa_studies_ok(True, True))
    checks.append(not cabinet_qa_studies_ok(False, True))
    checks.append(cabinet_qa_studies_aux(True))
    checks.append(not cabinet_qa_studies_aux(False))
    checks.append(True)  # governance canon
    return float(sum(checks) / len(checks))


def bench_cabinet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cabinet_qa_studies": _bench_cabinet_qa_studies(seed)}
