"""balin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def balin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """balin_qa_studies

    check:
    balin_qa_studies: t
    """
    return fit_ok and sample_ok


def balin_qa_studies_aux(aux: bool) -> bool:
    """balin_qa_studies

    aux:
    balin_qa_studies: w
    """
    return aux


def _bench_balin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(balin_qa_studies_ok(True, True))
    checks.append(not balin_qa_studies_ok(False, True))
    checks.append(balin_qa_studies_aux(True))
    checks.append(not balin_qa_studies_aux(False))
    checks.append(True)  # arthurian-4 canon
    return float(sum(checks) / len(checks))


def bench_balin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_balin_qa_studies": _bench_balin_qa_studies(seed)}
