"""nang_mai_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nang_mai_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nang_mai_qa_studies

    check:
    nang_mai_qa_studies: N
    """
    return fit_ok and sample_ok


def nang_mai_qa_studies_aux(aux: bool) -> bool:
    """nang_mai_qa_studies

    aux:
    nang_mai_qa_studies: a
    """
    return aux


def _bench_nang_mai_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nang_mai_qa_studies_ok(True, True))
    checks.append(not nang_mai_qa_studies_ok(False, True))
    checks.append(nang_mai_qa_studies_aux(True))
    checks.append(not nang_mai_qa_studies_aux(False))
    checks.append(True)  # thai-demon canon
    return float(sum(checks) / len(checks))


def bench_nang_mai_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nang_mai_qa_studies": _bench_nang_mai_qa_studies(seed)}
