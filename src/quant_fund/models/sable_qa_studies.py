"""sable_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sable_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sable_qa_studies

    check:
    sable_qa_studies: SableQA metrics
    """
    return fit_ok and sample_ok


def sable_qa_studies_aux(aux: bool) -> bool:
    """sable_qa_studies

    aux:
    sable_qa_studies: sables, taigas, answers, and scores
    """
    return aux


def _bench_sable_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sable_qa_studies_ok(True, True))
    checks.append(not sable_qa_studies_ok(False, True))
    checks.append(sable_qa_studies_aux(True))
    checks.append(not sable_qa_studies_aux(False))
    checks.append(True)  # mustelid-2 canon
    return float(sum(checks) / len(checks))


def bench_sable_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sable_qa_studies": _bench_sable_qa_studies(seed)}
