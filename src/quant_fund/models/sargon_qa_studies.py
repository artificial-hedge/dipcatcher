"""sargon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sargon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sargon_qa_studies

    check:
    sargon_qa_studies: SargonQA metrics
    """
    return fit_ok and sample_ok


def sargon_qa_studies_aux(aux: bool) -> bool:
    """sargon_qa_studies

    aux:
    sargon_qa_studies: sargon, river kings, answers, and scores
    """
    return aux


def _bench_sargon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sargon_qa_studies_ok(True, True))
    checks.append(not sargon_qa_studies_ok(False, True))
    checks.append(sargon_qa_studies_aux(True))
    checks.append(not sargon_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-2 canon
    return float(sum(checks) / len(checks))


def bench_sargon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sargon_qa_studies": _bench_sargon_qa_studies(seed)}
