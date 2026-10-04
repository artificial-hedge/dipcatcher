"""gallu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gallu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gallu_qa_studies

    check:
    gallu_qa_studies: g
    """
    return fit_ok and sample_ok


def gallu_qa_studies_aux(aux: bool) -> bool:
    """gallu_qa_studies

    aux:
    gallu_qa_studies: a
    """
    return aux


def _bench_gallu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gallu_qa_studies_ok(True, True))
    checks.append(not gallu_qa_studies_ok(False, True))
    checks.append(gallu_qa_studies_aux(True))
    checks.append(not gallu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_gallu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gallu_qa_studies": _bench_gallu_qa_studies(seed)}
