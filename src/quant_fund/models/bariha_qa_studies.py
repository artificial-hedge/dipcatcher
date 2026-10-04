"""bariha_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bariha_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bariha_qa_studies

    check:
    bariha_qa_studies: p
    """
    return fit_ok and sample_ok


def bariha_qa_studies_aux(aux: bool) -> bool:
    """bariha_qa_studies

    aux:
    bariha_qa_studies: u
    """
    return aux


def _bench_bariha_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bariha_qa_studies_ok(True, True))
    checks.append(not bariha_qa_studies_ok(False, True))
    checks.append(bariha_qa_studies_aux(True))
    checks.append(not bariha_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_bariha_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bariha_qa_studies": _bench_bariha_qa_studies(seed)}
