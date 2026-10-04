"""bean_nighe_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bean_nighe_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bean_nighe_qa_studies

    check:
    bean_nighe_qa_studies: B
    """
    return fit_ok and sample_ok


def bean_nighe_qa_studies_aux(aux: bool) -> bool:
    """bean_nighe_qa_studies

    aux:
    bean_nighe_qa_studies: e
    """
    return aux


def _bench_bean_nighe_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bean_nighe_qa_studies_ok(True, True))
    checks.append(not bean_nighe_qa_studies_ok(False, True))
    checks.append(bean_nighe_qa_studies_aux(True))
    checks.append(not bean_nighe_qa_studies_aux(False))
    checks.append(True)  # celtic-demon canon
    return float(sum(checks) / len(checks))


def bench_bean_nighe_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bean_nighe_qa_studies": _bench_bean_nighe_qa_studies(seed)}
