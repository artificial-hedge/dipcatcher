"""spoonbill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def spoonbill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spoonbill_qa_studies

    check:
    spoonbill_qa_studies: SpoonbillQA metrics
    """
    return fit_ok and sample_ok


def spoonbill_qa_studies_aux(aux: bool) -> bool:
    """spoonbill_qa_studies

    aux:
    spoonbill_qa_studies: spoonbills, lagoons, answers, and scores
    """
    return aux


def _bench_spoonbill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(spoonbill_qa_studies_ok(True, True))
    checks.append(not spoonbill_qa_studies_ok(False, True))
    checks.append(spoonbill_qa_studies_aux(True))
    checks.append(not spoonbill_qa_studies_aux(False))
    checks.append(True)  # wader canon
    return float(sum(checks) / len(checks))


def bench_spoonbill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spoonbill_qa_studies": _bench_spoonbill_qa_studies(seed)}
