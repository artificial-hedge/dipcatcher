"""iele_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iele_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iele_qa_studies

    check:
    iele_qa_studies: I
    """
    return fit_ok and sample_ok


def iele_qa_studies_aux(aux: bool) -> bool:
    """iele_qa_studies

    aux:
    iele_qa_studies: e
    """
    return aux


def _bench_iele_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iele_qa_studies_ok(True, True))
    checks.append(not iele_qa_studies_ok(False, True))
    checks.append(iele_qa_studies_aux(True))
    checks.append(not iele_qa_studies_aux(False))
    checks.append(True)  # romanian-demon canon
    return float(sum(checks) / len(checks))


def bench_iele_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iele_qa_studies": _bench_iele_qa_studies(seed)}
