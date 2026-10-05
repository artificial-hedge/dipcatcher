"""bune_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bune_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bune_qa_studies

    check:
    bune_qa_studies: B
    """
    return fit_ok and sample_ok


def bune_qa_studies_aux(aux: bool) -> bool:
    """bune_qa_studies

    aux:
    bune_qa_studies: u
    """
    return aux


def _bench_bune_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bune_qa_studies_ok(True, True))
    checks.append(not bune_qa_studies_ok(False, True))
    checks.append(bune_qa_studies_aux(True))
    checks.append(not bune_qa_studies_aux(False))
    checks.append(True)  # goetic-circle canon
    return float(sum(checks) / len(checks))


def bench_bune_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bune_qa_studies": _bench_bune_qa_studies(seed)}
