"""tambaran_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tambaran_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tambaran_qa_studies

    check:
    tambaran_qa_studies: T
    """
    return fit_ok and sample_ok


def tambaran_qa_studies_aux(aux: bool) -> bool:
    """tambaran_qa_studies

    aux:
    tambaran_qa_studies: a
    """
    return aux


def _bench_tambaran_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tambaran_qa_studies_ok(True, True))
    checks.append(not tambaran_qa_studies_ok(False, True))
    checks.append(tambaran_qa_studies_aux(True))
    checks.append(not tambaran_qa_studies_aux(False))
    checks.append(True)  # oceania-demon canon
    return float(sum(checks) / len(checks))


def bench_tambaran_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tambaran_qa_studies": _bench_tambaran_qa_studies(seed)}
