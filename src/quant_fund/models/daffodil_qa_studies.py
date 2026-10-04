"""daffodil_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def daffodil_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """daffodil_qa_studies

    check:
    daffodil_qa_studies: DaffodilQA metrics
    """
    return fit_ok and sample_ok


def daffodil_qa_studies_aux(aux: bool) -> bool:
    """daffodil_qa_studies

    aux:
    daffodil_qa_studies: daffodils, petals, answers, and scores
    """
    return aux


def _bench_daffodil_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(daffodil_qa_studies_ok(True, True))
    checks.append(not daffodil_qa_studies_ok(False, True))
    checks.append(daffodil_qa_studies_aux(True))
    checks.append(not daffodil_qa_studies_aux(False))
    checks.append(True)  # wildflower canon
    return float(sum(checks) / len(checks))


def bench_daffodil_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_daffodil_qa_studies": _bench_daffodil_qa_studies(seed)}
