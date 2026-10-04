"""dolly_varden_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dolly_varden_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dolly_varden_qa_studies

    check:
    dolly_varden_qa_studies: DollyVardenQA metrics
    """
    return fit_ok and sample_ok


def dolly_varden_qa_studies_aux(aux: bool) -> bool:
    """dolly_varden_qa_studies

    aux:
    dolly_varden_qa_studies: dolly vardens, cold streams, answers, and scores
    """
    return aux


def _bench_dolly_varden_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dolly_varden_qa_studies_ok(True, True))
    checks.append(not dolly_varden_qa_studies_ok(False, True))
    checks.append(dolly_varden_qa_studies_aux(True))
    checks.append(not dolly_varden_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_dolly_varden_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dolly_varden_qa_studies": _bench_dolly_varden_qa_studies(seed)}
