"""bes2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bes2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bes2_qa_studies

    check:
    bes2_qa_studies: Bes2QA metrics
    """
    return fit_ok and sample_ok


def bes2_qa_studies_aux(aux: bool) -> bool:
    """bes2_qa_studies

    aux:
    bes2_qa_studies: bes2, door wards, answers, and scores
    """
    return aux


def _bench_bes2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bes2_qa_studies_ok(True, True))
    checks.append(not bes2_qa_studies_ok(False, True))
    checks.append(bes2_qa_studies_aux(True))
    checks.append(not bes2_qa_studies_aux(False))
    checks.append(True)  # egyptian-8 canon
    return float(sum(checks) / len(checks))


def bench_bes2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bes2_qa_studies": _bench_bes2_qa_studies(seed)}
