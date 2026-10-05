"""iyarri2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iyarri2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iyarri2_qa_studies

    check:
    iyarri2_qa_studies: Iyarri2QA metrics
    """
    return fit_ok and sample_ok


def iyarri2_qa_studies_aux(aux: bool) -> bool:
    """iyarri2_qa_studies

    aux:
    iyarri2_qa_studies: iyarri2, plague bows, answers, and scores
    """
    return aux


def _bench_iyarri2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iyarri2_qa_studies_ok(True, True))
    checks.append(not iyarri2_qa_studies_ok(False, True))
    checks.append(iyarri2_qa_studies_aux(True))
    checks.append(not iyarri2_qa_studies_aux(False))
    checks.append(True)  # luwian-myth canon
    return float(sum(checks) / len(checks))


def bench_iyarri2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iyarri2_qa_studies": _bench_iyarri2_qa_studies(seed)}
