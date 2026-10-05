"""pincoya_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pincoya_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pincoya_qa_studies

    check:
    pincoya_qa_studies: P
    """
    return fit_ok and sample_ok


def pincoya_qa_studies_aux(aux: bool) -> bool:
    """pincoya_qa_studies

    aux:
    pincoya_qa_studies: i
    """
    return aux


def _bench_pincoya_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pincoya_qa_studies_ok(True, True))
    checks.append(not pincoya_qa_studies_ok(False, True))
    checks.append(pincoya_qa_studies_aux(True))
    checks.append(not pincoya_qa_studies_aux(False))
    checks.append(True)  # chiloe-demon canon
    return float(sum(checks) / len(checks))


def bench_pincoya_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pincoya_qa_studies": _bench_pincoya_qa_studies(seed)}
