"""knocker_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def knocker_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """knocker_qa_studies

    check:
    knocker_qa_studies: m
    """
    return fit_ok and sample_ok


def knocker_qa_studies_aux(aux: bool) -> bool:
    """knocker_qa_studies

    aux:
    knocker_qa_studies: i
    """
    return aux


def _bench_knocker_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(knocker_qa_studies_ok(True, True))
    checks.append(not knocker_qa_studies_ok(False, True))
    checks.append(knocker_qa_studies_aux(True))
    checks.append(not knocker_qa_studies_aux(False))
    checks.append(True)  # cornish-myth canon
    return float(sum(checks) / len(checks))


def bench_knocker_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knocker_qa_studies": _bench_knocker_qa_studies(seed)}
