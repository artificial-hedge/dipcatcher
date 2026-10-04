"""strix_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def strix_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """strix_qa_studies

    check:
    strix_qa_studies: s
    """
    return fit_ok and sample_ok


def strix_qa_studies_aux(aux: bool) -> bool:
    """strix_qa_studies

    aux:
    strix_qa_studies: t
    """
    return aux


def _bench_strix_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(strix_qa_studies_ok(True, True))
    checks.append(not strix_qa_studies_ok(False, True))
    checks.append(strix_qa_studies_aux(True))
    checks.append(not strix_qa_studies_aux(False))
    checks.append(True)  # european-vampire canon
    return float(sum(checks) / len(checks))


def bench_strix_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strix_qa_studies": _bench_strix_qa_studies(seed)}
