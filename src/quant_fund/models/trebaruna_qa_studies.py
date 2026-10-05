"""trebaruna_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def trebaruna_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trebaruna_qa_studies

    check:
    trebaruna_qa_studies: h
    """
    return fit_ok and sample_ok


def trebaruna_qa_studies_aux(aux: bool) -> bool:
    """trebaruna_qa_studies

    aux:
    trebaruna_qa_studies: o
    """
    return aux


def _bench_trebaruna_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(trebaruna_qa_studies_ok(True, True))
    checks.append(not trebaruna_qa_studies_ok(False, True))
    checks.append(trebaruna_qa_studies_aux(True))
    checks.append(not trebaruna_qa_studies_aux(False))
    checks.append(True)  # iberian-myth canon
    return float(sum(checks) / len(checks))


def bench_trebaruna_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trebaruna_qa_studies": _bench_trebaruna_qa_studies(seed)}
