"""ifa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ifa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ifa_qa_studies

    check:
    ifa_qa_studies: IfaQA metrics
    """
    return fit_ok and sample_ok


def ifa_qa_studies_aux(aux: bool) -> bool:
    """ifa_qa_studies

    aux:
    ifa_qa_studies: ifa, palm oracle, answers, and scores
    """
    return aux


def _bench_ifa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ifa_qa_studies_ok(True, True))
    checks.append(not ifa_qa_studies_ok(False, True))
    checks.append(ifa_qa_studies_aux(True))
    checks.append(not ifa_qa_studies_aux(False))
    checks.append(True)  # african-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_ifa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ifa_qa_studies": _bench_ifa_qa_studies(seed)}
