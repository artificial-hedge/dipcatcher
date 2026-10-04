"""hornbill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hornbill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hornbill_qa_studies

    check:
    hornbill_qa_studies: HornbillQA metrics
    """
    return fit_ok and sample_ok


def hornbill_qa_studies_aux(aux: bool) -> bool:
    """hornbill_qa_studies

    aux:
    hornbill_qa_studies: hornbills, figs, answers, and scores
    """
    return aux


def _bench_hornbill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hornbill_qa_studies_ok(True, True))
    checks.append(not hornbill_qa_studies_ok(False, True))
    checks.append(hornbill_qa_studies_aux(True))
    checks.append(not hornbill_qa_studies_aux(False))
    checks.append(True)  # canopybird canon
    return float(sum(checks) / len(checks))


def bench_hornbill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hornbill_qa_studies": _bench_hornbill_qa_studies(seed)}
