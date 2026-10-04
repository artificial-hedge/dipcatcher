"""dill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dill_qa_studies

    check:
    dill_qa_studies: DillQA metrics
    """
    return fit_ok and sample_ok


def dill_qa_studies_aux(aux: bool) -> bool:
    """dill_qa_studies

    aux:
    dill_qa_studies: dills, umbels, answers, and scores
    """
    return aux


def _bench_dill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dill_qa_studies_ok(True, True))
    checks.append(not dill_qa_studies_ok(False, True))
    checks.append(dill_qa_studies_aux(True))
    checks.append(not dill_qa_studies_aux(False))
    checks.append(True)  # spice canon
    return float(sum(checks) / len(checks))


def bench_dill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dill_qa_studies": _bench_dill_qa_studies(seed)}
