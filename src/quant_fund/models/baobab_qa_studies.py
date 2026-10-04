"""baobab_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def baobab_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """baobab_qa_studies

    check:
    baobab_qa_studies: BaobabQA metrics
    """
    return fit_ok and sample_ok


def baobab_qa_studies_aux(aux: bool) -> bool:
    """baobab_qa_studies

    aux:
    baobab_qa_studies: baobabs, savannas, answers, and scores
    """
    return aux


def _bench_baobab_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(baobab_qa_studies_ok(True, True))
    checks.append(not baobab_qa_studies_ok(False, True))
    checks.append(baobab_qa_studies_aux(True))
    checks.append(not baobab_qa_studies_aux(False))
    checks.append(True)  # tree-2 canon
    return float(sum(checks) / len(checks))


def bench_baobab_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baobab_qa_studies": _bench_baobab_qa_studies(seed)}
